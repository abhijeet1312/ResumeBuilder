import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import os
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Request model (remove prompt field)
class MatchRequest(BaseModel):
    resume: str
    job_desc: str
    prompt:str

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
def match_resume(resume, job_desc,prompt_template):
    # Define the fixed prompt template
    print(f"Received resume: ")  # Log the received resume

   
#     prompt_template = f"""
# You are a recruiter evaluating a candidate's resume against a job description.

# Job Description: {job_desc}

# Candidate Resume: {resume}

# Please assess both the technical skills (programming languages, tools, frameworks, relevant experience) and soft skills (design thinking, communication, adaptability, eagerness to learn, empathy, employee engagement). Take into account their education, certifications, and project experience.

# Provide a match score from 0 to 100 based on how well the candidate fits the role, giving realistic weight to key technical requirements and some consideration to soft skills and learning potential.

# First, return both the match score and a brief review.

# Then, extract the following fields from the resume and include them as key-value pairs directly in the same root-level object (do not nest them under any sub-object):

# - match_score
# - review
# - name
# - location
# - gender (male or female)
# - mobile
# - email
# - education (only the last university attended, the degree obtained, and the year of graduation — only the year, not the month)
# - last_company_worked_in
# - years_of_job_experience_after_graduation_in_months
# - current role of the candidate

# Note:
# - Use "years_of_job_experience_after_graduation_in_months" as the total number of full months worked after graduation.
# - If the candidate's graduation was in September 2024 and their first job started in January 2025, and the current month is April 2025, then the correct experience duration is 4 months.

# ⚠️ Strictly return a flat JSON object with no nested keys.
# ⚠️ Strictly give the current role of the candidate from the last experience in the JSON response.


# ✅ Only return a valid JSON response. Do not include any explanations, formatting, or text outside of the JSON.
# """

  

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

        # print(f"Cleaned Response: {result}")  # Log the cleaned response

        # Try parsing the response as JSON
        try:
            return json.loads(result)
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
        result = match_resume(request.resume, request.job_desc,request.prompt)
        
        # Return the result (either parsed JSON or raw response)
        return result

    except Exception as e:
        # Handle any errors during the process and return appropriate error message
        print(f"Error in /match/ endpoint: {e}")  # Log error for debugging
        raise HTTPException(status_code=500, detail=str(e))
