#model.py
import json
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import os
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from typing import List, Optional
print("jai mata di")

# Import the S3 functionality from file_upload_s3
from file_upload_s3 import (
    validate_pdf_file, 
    upload_file_to_s3, 
    delete_file_from_s3, 
    s3_config,
    ResumeUploadResponse,
    JobDescriptionUploadResponse
)

# Import the screening functions from endpoint2
from endpoint2 import screen_candidates_from_urls_logic, URLData

# Request model (remove prompt field)
class MatchRequest(BaseModel):
    resume: str
    job_desc: str
    prompt: str

app = FastAPI()
load_dotenv()  # Load from .env

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ✅ Allow all origins
    allow_credentials=False,  # ❌ Must be False when using allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embedding model for resume and job description matching
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Hugging Face Inference Client (ensure API key is set in environment variables)
client = InferenceClient(
    provider="novita",
    api_key=os.getenv("API_KEY"),  # Use environment variable for API key
)

# Initialize FAISS for similarity retrieval
dimension = 384
faiss_index = faiss.IndexFlatL2(dimension)
resumes = []

# Function to add a resume to FAISS index
def add_resume(resume_text):
    resume_vector = embedder.encode([resume_text])
    faiss_index.add(resume_vector)
    resumes.append(resume_text)

# Add sample resumes to the FAISS index
sample_resumes = [
    "Python Developer with 5 years experience in AI.",
    "Machine Learning Engineer skilled in TensorFlow and Deep Learning."
]
for r in sample_resumes:
    add_resume(r)

# Function to retrieve similar resumes based on job description
def retrieve_similar(job_desc, k=3):
    job_vector = embedder.encode([job_desc])
    _, indices = faiss_index.search(job_vector, k)
    return [resumes[i] for i in indices[0] if i < len(resumes)]

def match_resume(resume, job_desc, prompt_template):
    # Define the fixed prompt template
    print(f"Received resume: ")  # Log the received resume

    # Format the prompt with dynamic resume and job description
    prompt = prompt_template.format(resume=resume, job_desc=job_desc)
    print(f"Formatted Prompt: {prompt}")  # Log the formatted prompt to debug

    # Send the request to the AI model for matching
    try:
        completion = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.0
        )
        # Get and log the raw response
        result = completion.choices[0].message.content.strip()

        # Clean the result (remove extra newlines or unwanted characters)
        result = result.replace('\n', ' ').strip()
        

        # Try parsing the response as JSON
        try:
            parsed_result=json.loads(result)
            return {
               "test":"true",
                 "data":parsed_result
            }
            # return json.loads(result)
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            return {"error": "Failed to parse the response as JSON", "raw_response": result}
    
    except Exception as e:
        # If the AI model request fails, log and return the error
        print(f"Error in AI request: {e}")
        return {"error": "Failed to get AI response", "message": str(e)}

@app.post("/match/")
async def match(request: MatchRequest):
    try:
        print(f"Received Request: resume={request.resume}, job_desc={request.job_desc}")  # Log incoming request
        # Get the AI model's response, passing only resume and job description
        result = match_resume(request.resume, request.job_desc, request.prompt)
        
        # Return the result (either parsed JSON or raw response)
        return result

    except Exception as e:
        # Handle any errors during the process and return appropriate error message
        print(f"Error in /match/ endpoint: {e}")  # Log error for debugging
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/screen-candidates-from-urls/")
async def screen_candidates_from_urls(payload: URLData):
    """
    Screen candidates from S3 URLs - Route imported from endpoint2.py
    """
    return await screen_candidates_from_urls_logic(payload)

@app.post("/upload-resumes", response_model=ResumeUploadResponse)
async def upload_resumes(
    resumes: List[UploadFile] = File(..., description="Resume PDF files")
):
    """
    Upload resume PDF files to S3 bucket
    
    Args:
        resumes: List of resume PDF files
    
    Returns:
        JSON response with downloadable URLs for uploaded resume files
    """
    
    # Validate that files are provided
    if not resumes:
        raise HTTPException(
            status_code=400, 
            detail="Resume files are required"
        )
    
    # Validate file types
    for file in resumes:
        if not validate_pdf_file(file):
            raise HTTPException(
                status_code=400,
                detail=f"Only PDF files are allowed. Invalid file: {file.filename}"
            )
    
    # Check file size limits (5MB per file)
    max_file_size = 5 * 1024 * 1024  # 5MB
    for file in resumes:
        if file.size and file.size > max_file_size:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds 5MB limit: {file.filename}"
            )
    
    # Check file count limits
    if len(resumes) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 resume files allowed"
        )
    
    uploaded_resume_urls = []
    
    try:
        # Upload resume files
        for resume_file in resumes:
            # Reset file pointer to beginning
            await resume_file.seek(0)
            url = await upload_file_to_s3(resume_file, "resumes")
            uploaded_resume_urls.append(url)
        
        return ResumeUploadResponse(resumes=uploaded_resume_urls)
        
    except Exception as e:
        # Clean up uploaded files if there's an error
        print(f"Error during resume upload: {str(e)}")
        
        # Extract S3 keys from URLs and delete files
        for url in uploaded_resume_urls:
            try:
                # Extract S3 key from URL
                s3_key = url.split(f"https://{s3_config.bucket_name}.s3.{s3_config.region}.amazonaws.com/")[1]
                delete_file_from_s3(s3_key)
            except Exception as cleanup_error:
                print(f"Error during cleanup: {str(cleanup_error)}")
        
        raise HTTPException(status_code=500, detail=f"Resume upload failed: {str(e)}")

@app.post("/upload-job-descriptions", response_model=JobDescriptionUploadResponse)
async def upload_job_descriptions(
    job_description: UploadFile = File(..., description="Job description PDF file")
):
    """
    Upload a single job description PDF file to S3 bucket
    
    Args:
        job_description: Single job description PDF file
    
    Returns:
        JSON response with downloadable URL for uploaded job description file
    """
    
    # Validate that file is provided
    if not job_description:
        raise HTTPException(
            status_code=400, 
            detail="Job description file is required"
        )
    
    # Validate file type
    if not validate_pdf_file(job_description):
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF files are allowed. Invalid file: {job_description.filename}"
        )
    
    # Check file size limits (5MB per file)
    max_file_size = 5 * 1024 * 1024  # 5MB
    if job_description.size and job_description.size > max_file_size:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds 5MB limit: {job_description.filename}"
        )
    
    try:
        # Upload job description file
        await job_description.seek(0)
        url = await upload_file_to_s3(job_description, "job_descriptions")
        
        return JobDescriptionUploadResponse(job_descriptions=url)
        
    except Exception as e:
        # Clean up uploaded file if there's an error
        print(f"Error during job description upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Job description upload failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Resume and Job Description Upload API with AI Matching",
        "version": "1.0.0",
        "endpoints": {
            "upload-resumes": "/upload-resumes - POST - Upload resume PDF files",
            "upload-job-descriptions": "/upload-job-descriptions - POST - Upload a single job description PDF file",
            "match": "/match/ - POST - Match resume with job description",
            "screen-candidates-from-urls": "/screen-candidates-from-urls/ - POST - Screen candidates from S3 URLs",
            "health": "/health - GET - Health check",
            "docs": "/docs - GET - API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    # Environment variables required:
    # AWS_BUCKET_NAME, AWS_REGION, AWS_ACCESS_KEY, AWS_SECRET_KEY, API_KEY
    
    # FIXED: Get port from environment variable (Render provides PORT)
    port = int(os.environ.get("PORT", 8000))
    
    print("Starting FastAPI server...")
    print(f"Port: {port}")
    print(f"Bucket: {s3_config.bucket_name}")
    print(f"Region: {s3_config.region}")
    
    uvicorn.run(
        "model:app",  # Updated to match the filename
        host="0.0.0.0",
        port=port,  # FIXED: Use dynamic port from environment
        reload=False  # FIXED: Disable reload for production
    )


# #model.py
# import json
# from fastapi import FastAPI, HTTPException, File, UploadFile
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from huggingface_hub import InferenceClient
# import os
# from fastapi.middleware.cors import CORSMiddleware
# from dotenv import load_dotenv
# from typing import List, Optional
# print("jai mata di")

# # Import the S3 functionality from file_upload_s3
# from file_upload_s3 import (
#     validate_pdf_file, 
#     upload_file_to_s3, 
#     delete_file_from_s3, 
#     s3_config,
#     ResumeUploadResponse,
#     JobDescriptionUploadResponse
# )

# # Import the screening functions from endpoint2
# from endpoint2 import screen_candidates_from_urls_logic, URLData

# # Request model (remove prompt field)
# class MatchRequest(BaseModel):
#     resume: str
#     job_desc: str
#     prompt: str

# app = FastAPI()
# load_dotenv()  # Load from .env

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # ✅ Allow all origins
#     allow_credentials=False,  # ❌ Must be False when using allow_origins=["*"]
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load embedding model for resume and job description matching
# embedder = SentenceTransformer("all-MiniLM-L6-v2")

# # Hugging Face Inference Client (ensure API key is set in environment variables)
# client = InferenceClient(
#     provider="novita",
#     api_key=os.getenv("API_KEY"),  # Use environment variable for API key
# )

# # Initialize FAISS for similarity retrieval
# dimension = 384
# faiss_index = faiss.IndexFlatL2(dimension)
# resumes = []

# # Function to add a resume to FAISS index
# def add_resume(resume_text):
#     resume_vector = embedder.encode([resume_text])
#     faiss_index.add(resume_vector)
#     resumes.append(resume_text)

# # Add sample resumes to the FAISS index
# sample_resumes = [
#     "Python Developer with 5 years experience in AI.",
#     "Machine Learning Engineer skilled in TensorFlow and Deep Learning."
# ]
# for r in sample_resumes:
#     add_resume(r)

# # Function to retrieve similar resumes based on job description
# def retrieve_similar(job_desc, k=3):
#     job_vector = embedder.encode([job_desc])
#     _, indices = faiss_index.search(job_vector, k)
#     return [resumes[i] for i in indices[0] if i < len(resumes)]

# def match_resume(resume, job_desc, prompt_template):
#     # Define the fixed prompt template
#     print(f"Received resume: ")  # Log the received resume

#     # Format the prompt with dynamic resume and job description
#     prompt = prompt_template.format(resume=resume, job_desc=job_desc)
#     print(f"Formatted Prompt: {prompt}")  # Log the formatted prompt to debug

#     # Send the request to the AI model for matching
#     try:
#         completion = client.chat.completions.create(
#             model="mistralai/Mistral-7B-Instruct-v0.3",
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=1000,
#             temperature=0.0
#         )
#         # Get and log the raw response
#         result = completion.choices[0].message.content.strip()

#         # Clean the result (remove extra newlines or unwanted characters)
#         result = result.replace('\n', ' ').strip()
        

#         # Try parsing the response as JSON
#         try:
#             parsed_result=json.loads(result)
#             # return {
#             #     "test":"true",
#             #     "data":parsed_result
#             # }
#             return json.loads(result)
#         except json.JSONDecodeError as e:
#             print(f"JSON Decode Error: {e}")
#             return {"error": "Failed to parse the response as JSON", "raw_response": result}
    
#     except Exception as e:
#         # If the AI model request fails, log and return the error
#         print(f"Error in AI request: {e}")
#         return {"error": "Failed to get AI response", "message": str(e)}

# @app.post("/match/")
# async def match(request: MatchRequest):
#     try:
#         print(f"Received Request: resume={request.resume}, job_desc={request.job_desc}")  # Log incoming request
#         # Get the AI model's response, passing only resume and job description
#         result = match_resume(request.resume, request.job_desc, request.prompt)
        
#         # Return the result (either parsed JSON or raw response)
#         return result

#     except Exception as e:
#         # Handle any errors during the process and return appropriate error message
#         print(f"Error in /match/ endpoint: {e}")  # Log error for debugging
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/screen-candidates-from-urls/")
# async def screen_candidates_from_urls(payload: URLData):
#     """
#     Screen candidates from S3 URLs - Route imported from endpoint2.py
#     """
#     return await screen_candidates_from_urls_logic(payload)

# @app.post("/upload-resumes", response_model=ResumeUploadResponse)
# async def upload_resumes(
#     resumes: List[UploadFile] = File(..., description="Resume PDF files")
# ):
#     """
#     Upload resume PDF files to S3 bucket
    
#     Args:
#         resumes: List of resume PDF files
    
#     Returns:
#         JSON response with downloadable URLs for uploaded resume files
#     """
    
#     # Validate that files are provided
#     if not resumes:
#         raise HTTPException(
#             status_code=400, 
#             detail="Resume files are required"
#         )
    
#     # Validate file types
#     for file in resumes:
#         if not validate_pdf_file(file):
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Only PDF files are allowed. Invalid file: {file.filename}"
#             )
    
#     # Check file size limits (5MB per file)
#     max_file_size = 5 * 1024 * 1024  # 5MB
#     for file in resumes:
#         if file.size and file.size > max_file_size:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"File size exceeds 5MB limit: {file.filename}"
#             )
    
#     # Check file count limits
#     if len(resumes) > 10:
#         raise HTTPException(
#             status_code=400,
#             detail="Maximum 10 resume files allowed"
#         )
    
#     uploaded_resume_urls = []
    
#     try:
#         # Upload resume files
#         for resume_file in resumes:
#             # Reset file pointer to beginning
#             await resume_file.seek(0)
#             url = await upload_file_to_s3(resume_file, "resumes")
#             uploaded_resume_urls.append(url)
        
#         return ResumeUploadResponse(resumes=uploaded_resume_urls)
        
#     except Exception as e:
#         # Clean up uploaded files if there's an error
#         print(f"Error during resume upload: {str(e)}")
        
#         # Extract S3 keys from URLs and delete files
#         for url in uploaded_resume_urls:
#             try:
#                 # Extract S3 key from URL
#                 s3_key = url.split(f"https://{s3_config.bucket_name}.s3.{s3_config.region}.amazonaws.com/")[1]
#                 delete_file_from_s3(s3_key)
#             except Exception as cleanup_error:
#                 print(f"Error during cleanup: {str(cleanup_error)}")
        
#         raise HTTPException(status_code=500, detail=f"Resume upload failed: {str(e)}")

# @app.post("/upload-job-descriptions", response_model=JobDescriptionUploadResponse)
# async def upload_job_descriptions(
#     job_description: UploadFile = File(..., description="Job description PDF file")
# ):
#     """
#     Upload a single job description PDF file to S3 bucket
    
#     Args:
#         job_description: Single job description PDF file
    
#     Returns:
#         JSON response with downloadable URL for uploaded job description file
#     """
    
#     # Validate that file is provided
#     if not job_description:
#         raise HTTPException(
#             status_code=400, 
#             detail="Job description file is required"
#         )
    
#     # Validate file type
#     if not validate_pdf_file(job_description):
#         raise HTTPException(
#             status_code=400,
#             detail=f"Only PDF files are allowed. Invalid file: {job_description.filename}"
#         )
    
#     # Check file size limits (5MB per file)
#     max_file_size = 5 * 1024 * 1024  # 5MB
#     if job_description.size and job_description.size > max_file_size:
#         raise HTTPException(
#             status_code=400,
#             detail=f"File size exceeds 5MB limit: {job_description.filename}"
#         )
    
#     try:
#         # Upload job description file
#         await job_description.seek(0)
#         url = await upload_file_to_s3(job_description, "job_descriptions")
        
#         return JobDescriptionUploadResponse(job_descriptions=url)
        
#     except Exception as e:
#         # Clean up uploaded file if there's an error
#         print(f"Error during job description upload: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Job description upload failed: {str(e)}")

# @app.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     return {"status": "healthy", "message": "API is running"}

# @app.get("/")
# async def root():
#     """Root endpoint with API information"""
#     return {
#         "message": "Resume and Job Description Upload API with AI Matching",
#         "version": "1.0.0",
#         "endpoints": {
#             "upload-resumes": "/upload-resumes - POST - Upload resume PDF files",
#             "upload-job-descriptions": "/upload-job-descriptions - POST - Upload a single job description PDF file",
#             "match": "/match/ - POST - Match resume with job description",
#             "screen-candidates-from-urls": "/screen-candidates-from-urls/ - POST - Screen candidates from S3 URLs",
#             "health": "/health - GET - Health check",
#             "docs": "/docs - GET - API documentation"
#         }
#     }

# if __name__ == "__main__":
#     import uvicorn
#     # Environment variables required:
#     # AWS_BUCKET_NAME, AWS_REGION, AWS_ACCESS_KEY, AWS_SECRET_KEY, API_KEY
    
#     print("Starting FastAPI server...")
#     print(f"Bucket: {s3_config.bucket_name}")
#     print(f"Region: {s3_config.region}")
    
#     uvicorn.run(
#         "model:app",  # Updated to match the filename
#         host="0.0.0.0",
#         port=8000,
#         reload=True
#     )


# # #model.py
# # import json
# # from fastapi import FastAPI, HTTPException, File, UploadFile
# # from pydantic import BaseModel
# # import faiss
# # import numpy as np
# # from sentence_transformers import SentenceTransformer
# # from huggingface_hub import InferenceClient
# # import os
# # from fastapi.middleware.cors import CORSMiddleware
# # from dotenv import load_dotenv
# # from typing import List
# # print("jai mata di")

# # # Import the S3 functionality from file_upload_s3
# # from file_upload_s3 import (
# #     validate_pdf_file, 
# #     upload_file_to_s3, 
# #     delete_file_from_s3, 
# #     s3_config,
# #     ResumeUploadResponse,
# #     JobDescriptionUploadResponse
# # )

# # # Request model (remove prompt field)
# # class MatchRequest(BaseModel):
# #     resume: str
# #     job_desc: str
# #     prompt: str

# # app = FastAPI()
# # load_dotenv()  # Load from .env

# # # Add CORS middleware
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],  # ✅ Allow all origins
# #     allow_credentials=False,  # ❌ Must be False when using allow_origins=["*"]
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # # Load embedding model for resume and job description matching
# # embedder = SentenceTransformer("all-MiniLM-L6-v2")

# # # Hugging Face Inference Client (ensure API key is set in environment variables)
# # client = InferenceClient(
# #     provider="novita",
# #     api_key=os.getenv("API_KEY"),  # Use environment variable for API key
# # )

# # # Initialize FAISS for similarity retrieval
# # dimension = 384
# # faiss_index = faiss.IndexFlatL2(dimension)
# # resumes = []

# # # Function to add a resume to FAISS index
# # def add_resume(resume_text):
# #     resume_vector = embedder.encode([resume_text])
# #     faiss_index.add(resume_vector)
# #     resumes.append(resume_text)

# # # Add sample resumes to the FAISS index
# # sample_resumes = [
# #     "Python Developer with 5 years experience in AI.",
# #     "Machine Learning Engineer skilled in TensorFlow and Deep Learning."
# # ]
# # for r in sample_resumes:
# #     add_resume(r)

# # # Function to retrieve similar resumes based on job description
# # def retrieve_similar(job_desc, k=3):
# #     job_vector = embedder.encode([job_desc])
# #     _, indices = faiss_index.search(job_vector, k)
# #     return [resumes[i] for i in indices[0] if i < len(resumes)]

# # def match_resume(resume, job_desc, prompt_template):
# #     # Define the fixed prompt template
# #     print(f"Received resume: ")  # Log the received resume

# #     # Format the prompt with dynamic resume and job description
# #     prompt = prompt_template.format(resume=resume, job_desc=job_desc)
# #     print(f"Formatted Prompt: {prompt}")  # Log the formatted prompt to debug

# #     # Send the request to the AI model for matching
# #     try:
# #         completion = client.chat.completions.create(
# #             model="mistralai/Mistral-7B-Instruct-v0.3",
# #             messages=[{"role": "user", "content": prompt}],
# #             max_tokens=1000,
# #             temperature=0.0
# #         )
# #         # Get and log the raw response
# #         result = completion.choices[0].message.content.strip()

# #         # Clean the result (remove extra newlines or unwanted characters)
# #         result = result.replace('\n', ' ').strip()

# #         # Try parsing the response as JSON
# #         try:
# #             return json.loads(result)
# #         except json.JSONDecodeError as e:
# #             print(f"JSON Decode Error: {e}")
# #             return {"error": "Failed to parse the response as JSON", "raw_response": result}
    
# #     except Exception as e:
# #         # If the AI model request fails, log and return the error
# #         print(f"Error in AI request: {e}")
# #         return {"error": "Failed to get AI response", "message": str(e)}

# # @app.post("/match/")
# # async def match(request: MatchRequest):
# #     try:
# #         print(f"Received Request: resume={request.resume}, job_desc={request.job_desc}")  # Log incoming request
# #         # Get the AI model's response, passing only resume and job description
# #         result = match_resume(request.resume, request.job_desc, request.prompt)
        
# #         # Return the result (either parsed JSON or raw response)
# #         return result

# #     except Exception as e:
# #         # Handle any errors during the process and return appropriate error message
# #         print(f"Error in /match/ endpoint: {e}")  # Log error for debugging
# #         raise HTTPException(status_code=500, detail=str(e))

# # @app.post("/upload-resumes", response_model=ResumeUploadResponse)
# # async def upload_resumes(
# #     resumes: List[UploadFile] = File(..., description="Resume PDF files")
# # ):
# #     """
# #     Upload resume PDF files to S3 bucket
    
# #     Args:
# #         resumes: List of resume PDF files
    
# #     Returns:
# #         JSON response with downloadable URLs for uploaded resume files
# #     """
    
# #     # Validate that files are provided
# #     if not resumes:
# #         raise HTTPException(
# #             status_code=400, 
# #             detail="Resume files are required"
# #         )
    
# #     # Validate file types
# #     for file in resumes:
# #         if not validate_pdf_file(file):
# #             raise HTTPException(
# #                 status_code=400,
# #                 detail=f"Only PDF files are allowed. Invalid file: {file.filename}"
# #             )
    
# #     # Check file size limits (5MB per file)
# #     max_file_size = 5 * 1024 * 1024  # 5MB
# #     for file in resumes:
# #         if file.size and file.size > max_file_size:
# #             raise HTTPException(
# #                 status_code=400,
# #                 detail=f"File size exceeds 5MB limit: {file.filename}"
# #             )
    
# #     # Check file count limits
# #     if len(resumes) > 10:
# #         raise HTTPException(
# #             status_code=400,
# #             detail="Maximum 10 resume files allowed"
# #         )
    
# #     uploaded_resume_urls = []
    
# #     try:
# #         # Upload resume files
# #         for resume_file in resumes:
# #             # Reset file pointer to beginning
# #             await resume_file.seek(0)
# #             url = await upload_file_to_s3(resume_file, "resumes")
# #             uploaded_resume_urls.append(url)
        
# #         return ResumeUploadResponse(resumes=uploaded_resume_urls)
        
# #     except Exception as e:
# #         # Clean up uploaded files if there's an error
# #         print(f"Error during resume upload: {str(e)}")
        
# #         # Extract S3 keys from URLs and delete files
# #         for url in uploaded_resume_urls:
# #             try:
# #                 # Extract S3 key from URL
# #                 s3_key = url.split(f"https://{s3_config.bucket_name}.s3.{s3_config.region}.amazonaws.com/")[1]
# #                 delete_file_from_s3(s3_key)
# #             except Exception as cleanup_error:
# #                 print(f"Error during cleanup: {str(cleanup_error)}")
        
# #         raise HTTPException(status_code=500, detail=f"Resume upload failed: {str(e)}")

# # @app.post("/upload-job-descriptions", response_model=JobDescriptionUploadResponse)
# # async def upload_job_descriptions(
# #     job_description: UploadFile = File(..., description="Job description PDF file")
# # ):
# #     """
# #     Upload a single job description PDF file to S3 bucket
    
# #     Args:
# #         job_description: Single job description PDF file
    
# #     Returns:
# #         JSON response with downloadable URL for uploaded job description file
# #     """
    
# #     # Validate that file is provided
# #     if not job_description:
# #         raise HTTPException(
# #             status_code=400, 
# #             detail="Job description file is required"
# #         )
    
# #     # Validate file type
# #     if not validate_pdf_file(job_description):
# #         raise HTTPException(
# #             status_code=400,
# #             detail=f"Only PDF files are allowed. Invalid file: {job_description.filename}"
# #         )
    
# #     # Check file size limits (5MB per file)
# #     max_file_size = 5 * 1024 * 1024  # 5MB
# #     if job_description.size and job_description.size > max_file_size:
# #         raise HTTPException(
# #             status_code=400,
# #             detail=f"File size exceeds 5MB limit: {job_description.filename}"
# #         )
    
# #     try:
# #         # Upload job description file
# #         await job_description.seek(0)
# #         url = await upload_file_to_s3(job_description, "job_descriptions")
        
# #         return JobDescriptionUploadResponse(job_descriptions=url)
        
# #     except Exception as e:
# #         # Clean up uploaded file if there's an error
# #         print(f"Error during job description upload: {str(e)}")
# #         raise HTTPException(status_code=500, detail=f"Job description upload failed: {str(e)}")

# # @app.get("/health")
# # async def health_check():
# #     """Health check endpoint"""
# #     return {"status": "healthy", "message": "API is running"}

# # @app.get("/")
# # async def root():
# #     """Root endpoint with API information"""
# #     return {
# #         "message": "Resume and Job Description Upload API with AI Matching",
# #         "version": "1.0.0",
# #         "endpoints": {
# #             "upload-resumes": "/upload-resumes - POST - Upload resume PDF files",
# #             "upload-job-descriptions": "/upload-job-descriptions - POST - Upload a single job description PDF file",
# #             "match": "/match/ - POST - Match resume with job description",
# #             "health": "/health - GET - Health check",
# #             "docs": "/docs - GET - API documentation"
# #         }
# #     }

# # if __name__ == "__main__":
# #     import uvicorn
# #     # Environment variables required:
# #     # AWS_BUCKET_NAME, AWS_REGION, AWS_ACCESS_KEY, AWS_SECRET_KEY, API_KEY
    
# #     print("Starting FastAPI server...")
# #     print(f"Bucket: {s3_config.bucket_name}")
# #     print(f"Region: {s3_config.region}")
    
# #     uvicorn.run(
# #         "model:app",  # Updated to match the filename
# #         host="0.0.0.0",
# #         port=8000,
# #         reload=True
# #     )