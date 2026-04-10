"""Minimal FastAPI application for testing Railway connectivity."""

from fastapi import FastAPI

app = FastAPI(title="MWD Copilot Test", version="1.0.0")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hello from Railway!", "status": "healthy"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "mwd-copilot-test"}
