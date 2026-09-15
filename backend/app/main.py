# FastAPI application entry point for the SIF Precursor Detection platform
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.auth import get_current_user
from app.config import settings
from app.database import Base, SessionLocal, engine
from app.logging_config import configure_structured_logging
from app.migrations import (
    run_classification_state_migrations,
    run_feedback_migrations,
    run_ingestion_migrations,
    run_labeling_migrations,
    run_lifecycle_migrations,
    run_lsr_migrations,
    run_precursor_migrations,
    run_recommendation_migrations,
)
from app.models import User
from app.nlp.lsr import load_canonical_lsr_rules
from app.nlp.model import get_model_health_status, load_sif_model
from app.nlp.preprocess import verify_pii_detector
from app.routers import (
    admin,
    auth,
    clusters,
    dashboard,
    feedback,
    ingestion,
    recommendations,
    reports,
)
from app.schemas import PredictRequest, PredictResponse
from app.seed import seed_if_empty

configure_structured_logging()
logger = logging.getLogger(__name__)

# Local development initializes the schema once at module load. Compatibility
# migrations run in the lifespan handler below, so they are not executed twice.
# Production schema changes are applied by `alembic upgrade head` before the
# application starts; the API never performs DDL against production data.
if settings.app_env.lower() != "production":
    Base.metadata.create_all(bind=engine)

# Fail fast if canonical Life-Saving Rules config is missing or invalid
load_canonical_lsr_rules()

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize local schema/data dependencies without deprecated startup hooks."""
    if settings.app_env.lower() != "production":
        run_lsr_migrations(engine)
        run_recommendation_migrations(engine)
        run_ingestion_migrations(engine)
        run_lifecycle_migrations(engine)
        run_labeling_migrations(engine)
        run_feedback_migrations(engine)
        run_precursor_migrations(engine)
        run_classification_state_migrations(engine)
    db = SessionLocal()
    try:
        pii_mode = verify_pii_detector(require_spacy=settings.app_env.lower() == "production")
        logger.info("PII detector verified at startup", extra={"pii_detection_mode": pii_mode})
        seed_if_empty(db)

        model_data = load_sif_model()
        if model_data.get("pipeline") is None and settings.demo_mode:
            logger.info("No model found on startup. Triggering initial training.")
            from app.training import run_ml_training

            try:
                run_ml_training(db)
            except Exception as exc:  # startup must retain API availability in demo mode
                logger.error("Failed initial model training: %s", type(exc).__name__)
            load_sif_model()
        yield
    finally:
        db.close()


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

# Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    try:
        response: Response = await call_next(request)
    except Exception:
        logger.exception("Unhandled request failure", extra={"request_id": request_id})
        response = JSONResponse(status_code=500, content={"error": {"code": "INTERNAL_SERVER_ERROR", "message": "An unexpected server error occurred.", "request_id": request_id}})
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response

# CORS Middleware with strict origin control
origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if settings.app_env.lower() == "production" else (origins or ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router Registrations
app.include_router(auth.router, prefix=settings.api_prefix)
app.include_router(reports.router, prefix=settings.api_prefix)
app.include_router(clusters.router, prefix=settings.api_prefix)
app.include_router(dashboard.router, prefix=settings.api_prefix)
app.include_router(feedback.router, prefix=settings.api_prefix)
app.include_router(admin.router, prefix=settings.api_prefix)
app.include_router(recommendations.router, prefix=settings.api_prefix)
app.include_router(ingestion.router, prefix=settings.api_prefix)


# Global Exception Handler to sanitize unexpected production errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = request.headers.get("X-Request-ID", "unknown")
    logger.exception("Unhandled application exception", extra={"request_id": request_id})
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": {"code": "INTERNAL_SERVER_ERROR", "message": "An unexpected server error occurred.", "request_id": request_id}},
    )


@app.get("/health")
def liveness():
    """Liveness check: Is the application process alive?"""
    return {"status": "ok"}


@app.get("/health/readiness")
def readiness():
    """Readiness check: Can the application safely serve production requests?"""
    db_ok = False
    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception as e:
        logger.error(f"Database readiness check failed: {e}")
    finally:
        db.close()

    model_health = get_model_health_status()
    is_ready = db_ok and model_health["valid"]

    status_payload = {
        "status": "ready" if is_ready else "not_ready",
        "database": "ok" if db_ok else "failed",
        "model": "ok" if model_health["artifact_exists"] else "missing",
        "model_integrity": "valid" if model_health["valid"] else "failed",
        "environment": settings.app_env,
    }

    if not is_ready:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=status_payload)

    return status_payload


@app.get("/model-info")
def model_info(_: User = Depends(get_current_user)):
    """Model information endpoint: returns active version, calibration, threshold and integrity."""
    health = get_model_health_status()
    model_data = load_sif_model()
    opt_thresh = float(model_data.get("optimal_threshold", settings.sif_threshold))
    return {
        "model_version": model_data.get("model_version", "unknown"),
        "feature_version": model_data.get("feature_version", "tfidf-unigram-bigram-v1"),
        "preprocessing_version": model_data.get("preprocessing_version", "prep-pii-spell-abbr-v1"),
        "dataset_version": model_data.get("dataset_version", "sih-safety-ds-v1"),
        "data_provenance": model_data.get("data_provenance", health.get("data_provenance", "UNKNOWN_PROVENANCE")),
        "label_schema_version": model_data.get("label_schema_version", "sif-binary-v1"),
        "calibration_version": model_data.get("calibration_version", "uncalibrated-v0"),
        "calibration_method": model_data.get("calibration_method", "none"),
        "is_calibrated": model_data.get("is_calibrated", False),
        "threshold_version": model_data.get("threshold_version", "thresh-recall-prioritized-v1"),
        "threshold": opt_thresh,
        "optimal_threshold": opt_thresh,
        "training_run_id": model_data.get("training_run_id", ""),
        "trained_at": model_data.get("trained_at"),
        "model_type": "Calibrated TF-IDF + Logistic Regression (Platt Scaling)",
        "integrity_status": health["status"],
        "artifact_exists": health["artifact_exists"],
        "manifest_exists": health["manifest_exists"],
        "sha256": health.get("sha256"),
        "status": "active" if health["valid"] else "degraded",
    }


@app.post("/predict", response_model=PredictResponse)
def predict_adhoc(payload: PredictRequest, _: User = Depends(get_current_user)):
    """Return a prediction with only privacy-safe processed text, never raw input."""
    from app.nlp.model import predict_sif_details
    pred = predict_sif_details(payload.text)
    return PredictResponse(
        processed_text=pred["processed_text"],
        sif_probability=pred["sif_probability"],
        calibrated_sif_probability=pred.get("calibrated_sif_probability"),
        is_calibrated=pred.get("is_calibrated", False),
        calibration_status=pred.get("calibration_status", "UNCALIBRATED_FALLBACK"),
        sif_potential=pred["sif_potential"],
        classification_state=pred["classification_state"],
        requires_analyst_review=pred["requires_analyst_review"],
        uncertain_lower_threshold=settings.sif_uncertain_lower_threshold,
        sif_likely_threshold=settings.sif_likely_threshold,
        model_version=pred["model_version"],
        feature_version=pred["feature_version"],
        preprocessing_version=pred["preprocessing_version"],
        calibration_version=pred["calibration_version"],
        threshold_version=pred["threshold_version"],
        threshold=pred["threshold"],
        training_run_id=pred.get("training_run_id"),
    )
