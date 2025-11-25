import os

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from ..services.raml_service import analyze_apk_with_raml
from ..database import get_db
from ..auth import get_current_user
from .. import models, schemas
from celery.result import AsyncResult
from celery import states
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse
import markdown
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


router = APIRouter(prefix="/analysis", tags=["analysis"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_apk(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Upload and analyze an APK file. Requires valid JWT token."""
    # Get user email from authenticated user
    user = db.query(models.User).filter(models.User.username == current_user["username"]).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    user_email = user.email
    
    filename = file.filename or ""
    if not filename.endswith(".apk"):
        raise HTTPException(status_code=400, detail="Only APK files allowed")

    apk_path = os.path.join(UPLOAD_DIR, filename)
    with open(apk_path, "wb") as f:
        f.write(await file.read())

    # Submit the analysis task
    result = analyze_apk_with_raml.delay(apk_path, user_email)
    # Create APK report record in database with "Started" status
    apk_report = models.APKReport(
        user_email=user_email,
        apk_filename=filename,
        task_id=result.id,
        status="Started"
    )
    db.add(apk_report)
    db.commit()
    db.refresh(apk_report)
    
    return JSONResponse({"task_id": result.id, "status": "Task submitted", "report_id": apk_report.id})


@router.get("/reports")
async def get_user_reports(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Fetch all APK reports for the authenticated user. Requires valid JWT token."""
    # Get user email from authenticated user
    user = db.query(models.User).filter(models.User.username == current_user["username"]).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    user_email = user.email
    reports = db.query(models.APKReport).filter(models.APKReport.user_email == user_email).order_by(models.APKReport.created_at.desc()).all()
    
    # Enrich each report with current task status from Celery
    enriched_reports = []
    for report in reports:
        task_result = AsyncResult(report.task_id)
        
        # Update status based on Celery state
        if task_result.state == states.PENDING:
            setattr(report, "status", "Pending")
        elif task_result.state == states.STARTED:
            setattr(report, "status", "In Progress")
        elif task_result.state == states.SUCCESS:
            setattr(report, "status", "Completed")
            # task_result.result contains the analysis output; could store as markdown
        elif task_result.state == states.FAILURE:
            setattr(report, "status", "Failed")
        
        enriched_reports.append(schemas.APKReportResponse.from_orm(report))
    

    return JSONResponse(content=jsonable_encoder([report.dict() for report in enriched_reports]))


@router.get("/status/{task_id}")
async def get_analysis_status(
    task_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get analysis status for a specific task. Requires valid JWT token."""
    result = AsyncResult(task_id)
    
    # Fetch the report record to update its status
    report = db.query(models.APKReport).filter(models.APKReport.task_id == task_id).first()
    
    # Verify that the user can only access their own report
    if report:
        user = db.query(models.User).filter(models.User.username == current_user["username"]).first()
        if not user or user.email != report.user_email: # type: ignore
            raise HTTPException(status_code=403, detail="Forbidden: Cannot access another user's report")
    
    if result.state == states.PENDING:
        status_msg = "Pending"
    elif result.state == states.STARTED:
        status_msg = "In Progress"
    elif result.state == states.SUCCESS:
        status_msg = "Completed"
        # Store result/markdown in database if needed
        if report:
            report.markdown_report = str(result.result) # type: ignore
            setattr(report, "status", "Completed")
            db.commit()
    elif result.state == states.FAILURE:
        status_msg = "Failed"
        if report:
            setattr(report, "status", "Failed")
            db.commit()
    else:
        status_msg = result.state
    
    # Update report status in DB
    if report:
        setattr(report, "status", status_msg)
        db.commit()
    
    return JSONResponse({"status": status_msg, "result": result.result if result.state == states.SUCCESS else None})


@router.get("/report/{report_id}/download")
async def download_report(
    report_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    # Get report
    report = db.query(models.APKReport).filter(models.APKReport.id == report_id).first()

    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    # Only owner can download
    user = db.query(models.User).filter(models.User.username == current_user["username"]).first()
    if not user or user.email != report.user_email: # type: ignore
        raise HTTPException(status_code=403, detail="Forbidden")

    if not report.markdown_report: # type: ignore
        raise HTTPException(status_code=400, detail="Report not ready yet")

    output_path = f"downloads/report_{report_id}.pdf"
    os.makedirs("downloads", exist_ok=True)

    pdf = canvas.Canvas(output_path, pagesize=letter)
    text_object = pdf.beginText(40, 750)

    md_lines = report.markdown_report.split("\n")
    for line in md_lines:
        text_object.textLine(line)

    pdf.drawText(text_object)
    pdf.save()

    return FileResponse(
        output_path,
        media_type="application/pdf",
        filename=f"{report.apk_filename}_report.pdf"
    )