"""Health check endpoint."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "mwd-copilot-api"}


@router.get("/api/v1/health")
async def api_health_check():
    return {"status": "healthy", "service": "mwd-copilot-api", "version": "3.0.0"}
