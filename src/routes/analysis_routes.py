import os

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from ..services.raml_service import analyze_apk_with_raml
from celery.result import AsyncResult # type: ignore
from celery import states # type: ignore

router = APIRouter(prefix="/analysis", tags=["analysis"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_apk(file: UploadFile = File(...)):
    filename = file.filename or ""
    if not filename.endswith(".apk"):
        raise HTTPException(status_code=400, detail="Only APK files allowed")

    apk_path = os.path.join(UPLOAD_DIR, filename)
    with open(apk_path, "wb") as f:
        f.write(await file.read())

    result = analyze_apk_with_raml.delay(apk_path) # type: ignore
    return JSONResponse({"task_id": result.id, "status": "Task submitted"})


@router.get("/status/{task_id}")
async def get_analysis_status(task_id: str):
    result = AsyncResult(task_id)
    if result.state == states.PENDING:
        return JSONResponse({"status": "Pending"})
    elif result.state == states.STARTED:
        return JSONResponse({"status": "In Progress"})
    elif result.state == states.SUCCESS:
        return JSONResponse({"status": "Completed", "result": result.result})
    elif result.state == states.FAILURE:
        return JSONResponse({"status": "Failed", "error": str(result.result)})
    else:
        return JSONResponse({"status": result.state})
