# Add these imports to your existing main.py
import asyncio
from verification.verification_service import AquaChainVerificationService

# Add this after your existing imports and before app = FastAPI()
verification_service = AquaChainVerificationService()

from fastapi import FastAPI, UploadFile, File, Form, Depends
from fastapi import BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
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

load_dotenv()

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

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
        
        # Upload to Cloudinary (your existing code)
        photo_urls = []
        for photo in photos:
            result = cloudinary.uploader.upload(photo.file)
            photo_urls.append(result["secure_url"])
        
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
async def register(email: str = Form(...), username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    
    try:
        hashed_password = generate_password_hash(password, "pbkdf2:sha256", 8)

        user_check = db.query(User).filter(User.email==email).first()
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
        user = db.query(User).filter(User.email==email).first()

        if user and check_password_hash(user.password, password):
            return {"success": True, "message": "Login successful", "username": user.username}
        else:
            return {"success": False, "message": "Invalid username or password"}
    except Exception as e:
        return {"sucess": False, "message": f"Error: {e}"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000)

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
        approved_count = db.query(VerificationResult).filter(VerificationResult.verification_status == "APPROVED").count()
        rejected_count = db.query(VerificationResult).filter(VerificationResult.verification_status == "REJECTED").count()
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

# Add this new model after your existing Report and User models
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

# Add this line after Base.metadata.create_all(bind=engine)
# It's already there, just make sure the new table gets created
