# Add these imports to your existing main.py
import asyncio
from verification.verification_service import AquaChainVerificationService

# Add this after your existing imports and before app = FastAPI()
verification_service = AquaChainVerificationService()

from fastapi import FastAPI, UploadFile, File, Form, Depends
from fastapi import BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import json
import os
import uuid
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from fastapi.responses import HTMLResponse
import tempfile
from pathlib import Path

load_dotenv()

# Configure Cloudinary
# cloudinary.config(
#     cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
#     api_key=os.getenv("CLOUDINARY_API_KEY"),
#     api_secret=os.getenv("CLOUDINARY_API_SECRET")
# )

# Database setup with connection pooling
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./reports.db")
engine = create_engine(
    DATABASE_URL,
    pool_size=5,  # Number of connections to keep in pool
    max_overflow=10,  # Additional connections when pool is full
    pool_pre_ping=True,  # Test connections before use
    pool_recycle=3600,  # Recycle connections every hour
    echo=False  # Set to True for debugging SQL queries
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Database model
class Report(Base):
    __tablename__ = "reports"

    id = Column(String, primary_key=True)
    project_name = Column(String, nullable=False)
    photos = Column(Text)
    gps = Column(String, nullable=False)
    submitted_at = Column(DateTime, default=datetime.datetime.now)
    status = Column(String, default="Pending")


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    email = Column(String, nullable=False, unique=True)
    username = Column(String, nullable=False)
    password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.now)

class VerificationResult(Base):
    __tablename__ = "verification_results"

    id = Column(String, primary_key=True)
    report_id = Column(String, nullable=False)  # Links to your existing reports
    verification_status = Column(String, nullable=False)  # APPROVED, REJECTED, ERROR
    verification_score = Column(String, nullable=True)  # Store as string for JSON compatibility
    total_photos = Column(String, nullable=True)
    verified_photos = Column(String, nullable=True)
    rejected_photos = Column(String, nullable=True)
    detailed_results = Column(Text, nullable=True)  # Store full JSON result
    verified_at = Column(DateTime, default=datetime.datetime.now)

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# Database helper functions with dependency injection
def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def root():
    return {"message": "FastAPI backend is running on Render"}


@app.get("/api/reports")
async def get_reports(db: Session = Depends(get_db)):
    reports = db.query(Report).all()
    result = []
    for report in reports:
        result.append({
            "id": report.id,
            "projectName": report.project_name,
            "photos": json.loads(report.photos) if report.photos else [],
            "gps": report.gps,
            "submittedAt": report.submitted_at.isoformat() if report.submitted_at else None,
            "status": report.status
        })
    return {"success": True, "reports": result}


@app.post("/api/reports")
async def upload_report(
        background_tasks: BackgroundTasks,  # Add this import: from fastapi import BackgroundTasks
        projectName: str = Form(...),
        gps: str = Form(...),
        submittedAt: str = Form(...),
        photos: list[UploadFile] = File(...),
        db: Session = Depends(get_db)
):
    print(f"Received report: {projectName}, GPS: {gps}")

    try:
        # Generate unique ID for report
        report_id = str(uuid.uuid4())

        # Upload to Cloudinary (temporarily disabled for testing)
        photo_urls = []
        for photo in photos:
            # result = cloudinary.uploader.upload(photo.file)
            # photo_urls.append(result["secure_url"])
            photo_urls.append(f"dummy_url_{photo.filename}")  # For testing

        # Save to database (your existing code)
        new_report = Report(
            id=report_id,
            project_name=projectName,
            photos=json.dumps(photo_urls),
            gps=gps,
            submitted_at=datetime.datetime.fromisoformat(submittedAt.replace('Z', '+00:00'))
        )
        db.add(new_report)
        db.commit()

        # NEW: Start verification in background
        background_tasks.add_task(
            verify_report_async,
            report_id=report_id,
            photo_urls=photo_urls,
            project_name=projectName,
            gps=gps,
            db_session=db
        )

        return {"success": True, "id": report_id, "photo_urls": photo_urls, "verification_started": True}

    except Exception as e:
        print(f"Database error: {str(e)}")
        db.rollback()
        return {"success": False, "error": str(e)}


# Add this new background task function
async def verify_report_async(report_id: str, photo_urls: List[str], project_name: str, gps: str, db_session: Session):
    """
    Background task to run verification without blocking the upload response
    """
    try:
        # Run verification
        verification_result = await verification_service.verify_report_photos(
            report_id=report_id,
            cloudinary_urls=photo_urls,
            project_name=project_name,
            gps=gps
        )

        # Save verification result to database
        verification_record = VerificationResult(
            id=str(uuid.uuid4()),
            report_id=report_id,
            verification_status=verification_result.get('status', 'ERROR'),
            verification_score=str(verification_result.get('verification_score', 0)),
            total_photos=str(verification_result.get('total_photos', 0)),
            verified_photos=str(verification_result.get('verified_photos', 0)),
            rejected_photos=str(verification_result.get('rejected_photos', 0)),
            detailed_results=json.dumps(verification_result)
        )

        db_session.add(verification_record)
        db_session.commit()

        print(f"Verification completed for report {report_id}: {verification_result['status']}")

    except Exception as e:
        print(f"Background verification failed for {report_id}: {e}")


@app.post("/api/register")
async def register(email: str = Form(...), username: str = Form(...), password: str = Form(...),
                   db: Session = Depends(get_db)):
    try:
        hashed_password = generate_password_hash(password, "pbkdf2:sha256", 8)

        user_check = db.query(User).filter(User.email == email).first()
        if user_check:
            return {"success": False, "message": "User already exists"}

        user_id = str(uuid.uuid4())
        user = User(id=user_id, email=email, username=username, password=hashed_password)
        db.add(user)
        db.commit()
        return {"success": True, "message": "User registered successfully"}
    except Exception as e:
        return {"sucess": False, "message": f"Error: {e}"}


@app.post("/api/login")
async def login(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    try:
        user = db.query(User).filter(User.email == email).first()

        if user and check_password_hash(user.password, password):
            return {"success": True, "message": "Login successful", "username": user.username}
        else:
            return {"success": False, "message": "Invalid username or password"}
    except Exception as e:
        return {"sucess": False, "message": f"Error: {e}"}


@app.get("/verification", response_class=HTMLResponse)
async def verification_dashboard():
    """Level-1 verification web interface"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AquaChain Level-1 Verification</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .header { background: #2196F3; color: white; padding: 20px; border-radius: 5px; text-align: center; }
            .upload-area { background: white; padding: 30px; margin: 20px 0; border-radius: 5px; border: 2px dashed #ccc; text-align: center; }
            .button { background: #2196F3; color: white; border: none; padding: 12px 24px; border-radius: 4px; cursor: pointer; font-size: 16px; margin: 5px; }
            .button:hover { background: #1976D2; }
            .results { background: white; padding: 20px; margin: 20px 0; border-radius: 5px; }
            .approved { background: #4CAF50; color: white; padding: 10px; border-radius: 4px; }
            .rejected { background: #f44336; color: white; padding: 10px; border-radius: 4px; }
            .photo-result { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 4px; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🌊 AquaChain Level-1 Photo Verification</h1>
            <p>Upload images to test fraud detection</p>
        </div>

        <div class="upload-area">
            <h3>📸 Upload Photos for Verification</h3>
            <input type="file" id="fileInput" multiple accept="image/*">
            <br><br>
            <button class="button" onclick="verifyPhotos()">🔍 Run Verification</button>
        </div>

        <div id="results" class="results" style="display: none;"></div>

        <script>
            async function verifyPhotos() {
                const fileInput = document.getElementById('fileInput');
                const files = fileInput.files;

                if (files.length === 0) {
                    alert('Please select images first');
                    return;
                }

                const formData = new FormData();
                for (let file of files) {
                    formData.append('files', file);
                }

                document.getElementById('results').style.display = 'block';
                document.getElementById('results').innerHTML = '<h3>🔄 Processing...</h3>';

                try {
                    const response = await fetch('/api/verify-simple', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();
                    displayResults(data);
                } catch (error) {
                    document.getElementById('results').innerHTML = `<h3>❌ Error</h3><p>${error.message}</p>`;
                }
            }

            function displayResults(data) {
                let html = `
                    <h3>🔍 Verification Results</h3>
                    <div class="${data.overall_status.toLowerCase()}">
                        Overall Status: ${data.overall_status} 
                    </div>
                    <p><strong>Photos Processed:</strong> ${data.total_photos}</p>
                    <hr>
                `;

                data.photo_results.forEach(photo => {
                    const statusClass = photo.is_valid ? 'approved' : 'rejected';
                    html += `
                        <div class="photo-result">
                            <h4>${photo.filename}</h4>
                            <div class="${statusClass}">
                                ${photo.is_valid ? '✅ VERIFIED' : '❌ REJECTED'}
                            </div>
                            <p><strong>Reason:</strong> ${photo.reason}</p>
                            <p><strong>Size:</strong> ${photo.dimensions}</p>
                        </div>
                    `;
                });

                document.getElementById('results').innerHTML = html;
            }
        </script>
    </body>
    </html>
    """


@app.post("/api/verify-simple")
async def verify_photos_simple(files: list[UploadFile] = File(...)):
    """Simple verification without complex dependencies"""
    if not files:
        return {"error": "No files provided"}

    photo_results = []
    valid_count = 0

    for file in files:
        if not file.content_type.startswith('image/'):
            continue

        # Read file content
        content = await file.read()
        file_size = len(content)

        # Simple checks without OpenCV
        is_too_small = file_size < 10000  # Less than 10KB probably invalid
        is_too_large = file_size > 50000000  # More than 50MB probably invalid

        # Basic filename checks
        suspicious_filename = any(word in file.filename.lower() for word in ['screenshot', 'screen', 'capture'])

        is_valid = not (is_too_small or is_too_large or suspicious_filename)

        reason = "Valid image"
        if is_too_small:
            reason = "File too small"
        elif is_too_large:
            reason = "File too large"
        elif suspicious_filename:
            reason = "Suspicious filename (possible screenshot)"

        photo_results.append({
            "filename": file.filename,
            "is_valid": is_valid,
            "reason": reason,
            "file_size": file_size,
            "dimensions": f"{file_size} bytes"
        })

        if is_valid:
            valid_count += 1

    overall_status = "APPROVED" if valid_count > len(photo_results) / 2 else "REJECTED"

    return {
        "overall_status": overall_status,
        "total_photos": len(photo_results),
        "valid_photos": valid_count,
        "photo_results": photo_results
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8001)


@app.get("/api/reports/{report_id}/verification")
async def get_report_verification(report_id: str, db: Session = Depends(get_db)):
    """
    Get verification status for a specific report
    """
    verification = db.query(VerificationResult).filter(VerificationResult.report_id == report_id).first()

    if not verification:
        return {"success": False, "message": "Verification not found or still processing"}

    return {
        "success": True,
        "verification": {
            "status": verification.verification_status,
            "score": verification.verification_score,
            "total_photos": verification.total_photos,
            "verified_photos": verification.verified_photos,
            "rejected_photos": verification.rejected_photos,
            "verified_at": verification.verified_at.isoformat() if verification.verified_at else None
        }
    }


@app.get("/api/reports/{report_id}/verification/details")
async def get_detailed_verification(report_id: str, db: Session = Depends(get_db)):
    """
    Get detailed verification results including HTML report
    """
    verification = db.query(VerificationResult).filter(VerificationResult.report_id == report_id).first()

    if not verification:
        return {"success": False, "message": "Verification not found"}

    try:
        detailed_results = json.loads(verification.detailed_results)

        # Generate HTML report using your existing PhotoVerifier
        html_report = verification_service.verifier.generate_verification_report(detailed_results)

        return {
            "success": True,
            "detailed_results": detailed_results,
            "html_report": html_report
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/verification/summary")
async def get_verification_summary(db: Session = Depends(get_db)):
    """
    Get overall verification statistics for dashboard
    """
    try:
        total_verifications = db.query(VerificationResult).count()
        approved_count = db.query(VerificationResult).filter(
            VerificationResult.verification_status == "APPROVED").count()
        rejected_count = db.query(VerificationResult).filter(
            VerificationResult.verification_status == "REJECTED").count()
        pending_count = total_verifications - approved_count - rejected_count

        return {
            "success": True,
            "summary": {
                "total_verifications": total_verifications,
                "approved_count": approved_count,
                "rejected_count": rejected_count,
                "pending_count": pending_count,
                "approval_rate": (approved_count / total_verifications * 100) if total_verifications > 0 else 0
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}