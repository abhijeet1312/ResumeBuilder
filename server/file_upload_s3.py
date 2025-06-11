import os
import uuid
from datetime import datetime
from typing import List
import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Resume and Job Description Upload API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response model
class UploadResponse(BaseModel):
    resumes: List[str]
    job_descriptions: List[str]

# AWS S3 Configuration
class S3Config:
    def __init__(self):
        self.bucket_name = os.getenv("AWS_BUCKET_NAME")
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.access_key = os.getenv("AWS_ACCESS_KEY")
        self.secret_key = os.getenv("AWS_SECRET_KEY")
        
        if not all([self.bucket_name, self.access_key, self.secret_key]):
            raise ValueError("Missing required AWS credentials in environment variables")
        
        self.s3_client = boto3.client(
            's3',
            region_name=self.region,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        )

s3_config = S3Config()

def validate_pdf_file(file: UploadFile) -> bool:
    """Validate if the uploaded file is a PDF"""
    if file.content_type != "application/pdf":
        return False
    return True

def generate_unique_filename(original_filename: str) -> str:
    """Generate a unique filename using timestamp and UUID"""
    timestamp = int(datetime.now().timestamp() * 1000)
    unique_id = str(uuid.uuid4())[:8]
    name, ext = os.path.splitext(original_filename)
    return f"{timestamp}_{unique_id}_{name}{ext}"

async def upload_file_to_s3(file: UploadFile, folder: str) -> str:
    """Upload a file to S3 and return the public URL"""
    try:
        # Generate unique filename
        unique_filename = generate_unique_filename(file.filename)
        s3_key = f"{folder}/{unique_filename}"
        
        # Read file content
        file_content = await file.read()
        
        # Upload to S3
        s3_config.s3_client.put_object(
            Bucket=s3_config.bucket_name,
            Key=s3_key,
            Body=file_content,
            ContentType=file.content_type,
            ACL='private'  # Make file publicly accessible
        )
        
        # Generate public URL
        public_url = f"https://{s3_config.bucket_name}.s3.{s3_config.region}.amazonaws.com/{s3_key}"
        
        return public_url
        
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"Error uploading to S3: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

def delete_file_from_s3(s3_key: str):
    """Delete a file from S3 bucket"""
    try:
        s3_config.s3_client.delete_object(
            Bucket=s3_config.bucket_name,
            Key=s3_key
        )
        print(f"Deleted {s3_key} from bucket {s3_config.bucket_name}")
    except ClientError as e:
        print(f"Error deleting file {s3_key}: {str(e)}")

@app.post("/upload", response_model=UploadResponse)
async def upload_files(
    resumes: List[UploadFile] = File(..., description="Resume PDF files"),
    job_descriptions: List[UploadFile] = File(..., description="Job description PDF files")
):
    """
    Upload resume and job description PDF files to S3 bucket
    
    Args:
        resumes: List of resume PDF files
        job_descriptions: List of job description PDF files
    
    Returns:
        JSON response with downloadable URLs for uploaded files
    """
    
    # Validate that files are provided
    if not resumes or not job_descriptions:
        raise HTTPException(
            status_code=400, 
            detail="Both resumes and job description files are required"
        )
    
    # Validate file types
    for file in resumes + job_descriptions:
        if not validate_pdf_file(file):
            raise HTTPException(
                status_code=400,
                detail=f"Only PDF files are allowed. Invalid file: {file.filename}"
            )
    
    # Check file size limits (5MB per file)
    max_file_size = 5 * 1024 * 1024  # 5MB
    for file in resumes + job_descriptions:
        if file.size and file.size > max_file_size:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds 5MB limit: {file.filename}"
            )
    
    # Check file count limits
    if len(resumes) > 10 or len(job_descriptions) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 files allowed per category"
        )
    
    uploaded_resume_urls = []
    uploaded_job_desc_urls = []
    
    try:
        # Upload resume files
        for resume_file in resumes:
            # Reset file pointer to beginning
            await resume_file.seek(0)
            url = await upload_file_to_s3(resume_file, "resumes")
            uploaded_resume_urls.append(url)
        
        # Upload job description files
        for job_desc_file in job_descriptions:
            # Reset file pointer to beginning
            await job_desc_file.seek(0)
            url = await upload_file_to_s3(job_desc_file, "job_descriptions")
            uploaded_job_desc_urls.append(url)
        
        return UploadResponse(
            resumes=uploaded_resume_urls,
            job_descriptions=uploaded_job_desc_urls
        )
        
    except Exception as e:
        # Clean up uploaded files if there's an error
        print(f"Error during upload: {str(e)}")
        
        # Extract S3 keys from URLs and delete files
        all_uploaded_urls = uploaded_resume_urls + uploaded_job_desc_urls
        for url in all_uploaded_urls:
            try:
                # Extract S3 key from URL
                s3_key = url.split(f"https://{s3_config.bucket_name}.s3.{s3_config.region}.amazonaws.com/")[1]
                delete_file_from_s3(s3_key)
            except Exception as cleanup_error:
                print(f"Error during cleanup: {str(cleanup_error)}")
        
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Resume and Job Description Upload API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload - POST - Upload resumes and job descriptions",
            "health": "/health - GET - Health check",
            "docs": "/docs - GET - API documentation"
        }
    }

if __name__ == "__main__":
    # Environment variables required:
    # AWS_BUCKET_NAME, AWS_REGION, AWS_ACCESS_KEY, AWS_SECRET_KEY
    
    print("Starting FastAPI server...")
    print(f"Bucket: {s3_config.bucket_name}")
    print(f"Region: {s3_config.region}")
    
    uvicorn.run(
        "main:app",  # Change this to your actual filename if different
        host="0.0.0.0",
        port=8000,
        reload=True
    )