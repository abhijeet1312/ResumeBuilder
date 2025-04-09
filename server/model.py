from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

app = FastAPI()
load_dotenv()  # Load from .env
print("jai mata di")

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Hugging Face Inference Client
client = InferenceClient(
    provider="novita",
    api_key=os.getenv("API_KEY"),  # Use environment variable for API key
)

# Initialize FAISS for retrieval
dimension = 384
faiss_index = faiss.IndexFlatL2(dimension)
resumes = []

# Add resumes to FAISS
def add_resume(resume_text):
    resume_vector = embedder.encode([resume_text])
    faiss_index.add(resume_vector)
    resumes.append(resume_text)

sample_resumes = [
    "Python Developer with 5 years experience in AI.",
    "Machine Learning Engineer skilled in TensorFlow and Deep Learning."
]
for r in sample_resumes:
    add_resume(r)

# Retrieve similar resumes
def retrieve_similar(job_desc, k=3):
    job_vector = embedder.encode([job_desc])
    _, indices = faiss_index.search(job_vector, k)
    return [resumes[i] for i in indices[0] if i < len(resumes)]

# AI Matching Function
def match_resume(resume, job_desc, prompt_template):
    # Format the prompt with dynamic inputs
    prompt = prompt_template.format(resume=resume, job_desc=job_desc)

    completion = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.0
    )

    # print(completion.choices[0].message.content)
    return completion.choices[0].message.content

# Request model
class MatchRequest(BaseModel):
    resume: str
    job_desc: str
    prompt: str  # Accept dynamic prompt

@app.post("/match/")
async def match(request: MatchRequest):
    try:
        result = match_resume(request.resume, request.job_desc, request.prompt)
        clean_result = result.strip()

        # Extract score and review
        if "matchscore" in clean_result.lower() and "review" in clean_result.lower():
            parts = clean_result.lower().split("matchscore:")[1].split("review:")
            score_str = parts[0].strip().replace("%", "").replace(",", "")
            review_text = parts[1].strip()

            score = int(score_str)

            return {
                "match_score": score,
                "review": review_text
            }
        else:
            return {"raw_result": clean_result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


