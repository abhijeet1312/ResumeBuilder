from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse

from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import tempfile
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from screening import CandidateScreeningAgent
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

app = FastAPI(
    title="Resume Screening API",
    description="API for screening job candidates using AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ✅ Allow all origins
    allow_credentials=False,  # ❌ Must be False when using allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a temporary directory for storing uploaded files
TEMP_DIR = Path("temp_resumes")
TEMP_DIR.mkdir(exist_ok=True)

class ScreeningResponse:
    """Response model for screening results"""
    def __init__(self, success: bool, message: str, data: dict = None, error: str = None):
        self.success = success
        self.message = message
        self.data = data or {}
        self.error = error

def extract_text_from_file(file_path: str) -> str:
    """
    Extract text from uploaded file (PDF, DOCX, or TXT)
    
    Args:
        file_path: Path to the file
        
    Returns:
        Extracted text content
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            return "\n".join([doc.page_content for doc in documents])
        
        elif file_extension == '.docx':
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            return "\n".join([doc.page_content for doc in documents])
        
        elif file_extension == '.txt':
            loader = TextLoader(file_path)
            documents = loader.load()
            return "\n".join([doc.page_content for doc in documents])
        
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
            
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Error extracting text from {file_extension} file: {str(e)}"
        )

@app.post("/screen-candidates/")
async def screen_candidates(
    job_description_file: UploadFile = File(..., description="Job description file (PDF, DOCX, TXT)"),
    voice_interview_threshold: float = Form(3.0, description="Minimum score threshold for voice interviews"),
    resumes: List[UploadFile] = File(..., description="Resume files (PDF, DOCX, TXT)")
):
    """
    Screen candidates based on their resumes and job description file.
    
    Args:
        job_description_file: Job description uploaded as a file (PDF, DOCX, TXT)
        voice_interview_threshold: Minimum score for voice interview qualification
        resumes: List of resume files to screen
        
    Returns:
        JSON response with screening results
    """
    
    saved_files = []
    job_desc_file_path = None
    
    try:
        # Validate file types
        allowed_extensions = {'.pdf', '.docx', '.txt'}
        
        # Validate and save job description file
        if not job_description_file.filename:
            raise HTTPException(status_code=400, detail="Job description filename is empty")
        
        job_desc_extension = Path(job_description_file.filename).suffix.lower()
        if job_desc_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported job description file type: {job_desc_extension}. Supported types: {', '.join(allowed_extensions)}"
            )
        
        # Save job description file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=job_desc_extension) as temp_job_file:
            job_content = await job_description_file.read()
            temp_job_file.write(job_content)
            job_desc_file_path = temp_job_file.name
        
        # Extract job description text
        print(f"Extracting job description from: {job_description_file.filename}")
        job_description_text = extract_text_from_file(job_desc_file_path)
        
        if not job_description_text.strip():
            raise HTTPException(
                status_code=400, 
                detail="Job description file appears to be empty or text could not be extracted"
            )
        
        print(f"Job description extracted successfully. Length: {len(job_description_text)} characters")
        
        # Save uploaded resume files temporarily
        for resume in resumes:
            if not resume.filename:
                raise HTTPException(status_code=400, detail="Empty resume filename provided")
            
            file_extension = Path(resume.filename).suffix.lower()
            if file_extension not in allowed_extensions:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported resume file type: {file_extension}. Supported types: {', '.join(allowed_extensions)}"
                )
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                content = await resume.read()
                temp_file.write(content)
                saved_files.append(temp_file.name)
        
        print(f"Saved {len(saved_files)} resume files: {[os.path.basename(f) for f in saved_files]}")
        
        # Initialize the screening agent with extracted job description
        agent = CandidateScreeningAgent(job_description=job_description_text)
        
        # Screen all candidates
        print("Starting candidate screening...")
        assessments = agent.batch_screen_candidates(saved_files)
        
        # Generate report and get qualified candidates
        report_path = TEMP_DIR / "candidate_assessments.csv"
        qualified_data = agent.generate_report(
            assessments, 
            output_path=str(report_path),
            voice_interview_threshold=voice_interview_threshold
        )
        
        # Prepare response data
        response_data = {
            "timestamp": datetime.now().isoformat(),
            "job_description_file": job_description_file.filename,
            "job_description_preview": job_description_text[:500] + "..." if len(job_description_text) > 500 else job_description_text,
            "job_description_length": len(job_description_text),
            "total_candidates": len(assessments),
            "assessments": assessments,
            "qualified_candidates": qualified_data.get('qualified_candidates', []),
            "total_qualified": qualified_data.get('total_qualified', 0),
            "email_recipients": qualified_data.get('email_recipients', []),
            "voice_interview_threshold": voice_interview_threshold,
            "report_generated": True
        }
        
        # Trigger voice interviews for qualified candidates
        voice_results = None
        if qualified_data.get('qualified_candidates'):
            try:
                print("Triggering voice interviews for qualified candidates...")
                voice_results = agent.trigger_voice_interviews_for_qualified(qualified_data)
                response_data["voice_interviews"] = {
                    "initiated": True,
                    "results": voice_results,
                    "completed_screenings": voice_results.get('completed_screenings', 0) if voice_results else 0,
                    "qualified_count": voice_results.get('qualified_count', 0) if voice_results else 0
                }
            except Exception as voice_error:
                print(f"Voice interview error: {voice_error}")
                response_data["voice_interviews"] = {
                    "initiated": False,
                    "error": str(voice_error)
                }
        else:
            response_data["voice_interviews"] = {
                "initiated": False,
                "reason": "No candidates qualified for voice interviews"
            }
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Screening completed successfully",
                "data": response_data
            }
        )
        
    except HTTPException as he:
        # Re-raise HTTP exceptions as-is
        raise he
        
    except Exception as e:
        print(f"Unexpected error during screening: {e}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Screening failed",
                "error": str(e)
            }
        )
    finally:
        # Clean up temporary files
        for file_path in saved_files:
            try:
                os.unlink(file_path)
            except Exception as cleanup_error:
                print(f"Error cleaning up resume file {file_path}: {cleanup_error}")
        
        # Clean up job description file
        if job_desc_file_path:
            try:
                os.unlink(job_desc_file_path)
            except Exception as cleanup_error:
                print(f"Error cleaning up job description file {job_desc_file_path}: {cleanup_error}")

@app.post("/screen-candidates-text/")
async def screen_candidates_with_text_job_desc(
    job_description: str = Form(..., description="Job description as text"),
    voice_interview_threshold: float = Form(3.0, description="Minimum score threshold for voice interviews"),
    resumes: List[UploadFile] = File(..., description="Resume files (PDF, DOCX, TXT)")
):
    """
    Alternative endpoint that accepts job description as text (backward compatibility).
    
    Args:
        job_description: The job requirements and description as text
        voice_interview_threshold: Minimum score for voice interview qualification
        resumes: List of resume files to screen
        
    Returns:
        JSON response with screening results
    """
    
    saved_files = []
    
    try:
        # Validate file types
        allowed_extensions = {'.pdf', '.docx', '.txt'}
        
        # Save uploaded files temporarily
        for resume in resumes:
            if not resume.filename:
                raise HTTPException(status_code=400, detail="Empty filename provided")
            
            file_extension = Path(resume.filename).suffix.lower()
            if file_extension not in allowed_extensions:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type: {file_extension}. Supported types: {', '.join(allowed_extensions)}"
                )
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                content = await resume.read()
                temp_file.write(content)
                saved_files.append(temp_file.name)
        
        # Initialize the screening agent
        print(saved_files)
        agent = CandidateScreeningAgent(job_description=job_description)
        
        # Screen all candidates
        assessments = agent.batch_screen_candidates(saved_files)
        
        # Generate report and get qualified candidates
        report_path = TEMP_DIR / "candidate_assessments.csv"
        qualified_data = agent.generate_report(
            assessments, 
            output_path=str(report_path),
            voice_interview_threshold=voice_interview_threshold
        )
        
        # Prepare response data
        response_data = {
            "timestamp": datetime.now().isoformat(),
            "job_description": job_description,
            "total_candidates": len(assessments),
            "assessments": assessments,
            "qualified_candidates": qualified_data.get('qualified_candidates', []),
            "total_qualified": qualified_data.get('total_qualified', 0),
            "email_recipients": qualified_data.get('email_recipients', []),
            "voice_interview_threshold": voice_interview_threshold,
            "report_generated": True
        }
        
        # Trigger voice interviews for qualified candidates
        voice_results = None
        if qualified_data.get('qualified_candidates'):
            try:
                voice_results = agent.trigger_voice_interviews_for_qualified(qualified_data)
                response_data["voice_interviews"] = {
                    "initiated": True,
                    "results": voice_results,
                    "completed_screenings": voice_results.get('completed_screenings', 0) if voice_results else 0,
                    "qualified_count": voice_results.get('qualified_count', 0) if voice_results else 0
                }
            except Exception as voice_error:
                response_data["voice_interviews"] = {
                    "initiated": False,
                    "error": str(voice_error)
                }
        else:
            response_data["voice_interviews"] = {
                "initiated": False,
                "reason": "No candidates qualified for voice interviews"
            }
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Screening completed successfully",
                "data": response_data
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Screening failed",
                "error": str(e)
            }
        )
    finally:
        # Clean up temporary files
        for file_path in saved_files:
            try:
                os.unlink(file_path)
            except:
                pass

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    print("🚀 Resume Screening API started successfully!")
    print(f"📁 Temporary files will be stored in: {TEMP_DIR.absolute()}")
    print("📋 Available endpoints:")
    print("  - POST /screen-candidates/ (with PDF job description)")
    print("  - POST /screen-candidates-text/ (with text job description)")
    print("  - GET /health")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    print("🛑 Resume Screening API shutting down...")
