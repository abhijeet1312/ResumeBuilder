import os
import re
import tempfile
import traceback
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
from jai import send_bulk_email 
print(API_KEY)

from huggingface_hub import InferenceClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field, field_validator

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import pandas as pd
load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
print(api_token)

from getpass import getpass

# HUGGINGFACEHUB_API_TOKEN = getpass()
import os

class CandidateAssessment(BaseModel):
    """Model for structured output of candidate assessment"""
    candidate_name: str = Field(description="The name of the candidate")
    skills_match_score: int = Field(description="Score from 1-10 on how well the candidate's skills match the requirements")
    experience_relevance_score: int = Field(description="Score from 1-10 on the relevance of candidate's experience")
    education_match_score: int = Field(description="Score from 1-10 on educational qualification match")
    overall_fit_score: int = Field(description="Score from 1-10 on overall fitness for the role")
    strengths: List[str] = Field(description="List of candidate's key strengths")
    weaknesses: List[str] = Field(description="List of candidate's key weaknesses")
    recommendation: str = Field(description="Short recommendation: 'Strong Match', 'Potential Match', or 'Not Recommended'")
    candidate_email:str = Field("The email of the candidate")
    
    @field_validator('skills_match_score', 'experience_relevance_score', 'education_match_score', 'overall_fit_score')
    def score_must_be_valid(cls, v):
        if not 1 <= v <= 10:
            raise ValueError('Score must be between 1 and 10')
        return v
    
    @field_validator('recommendation')
    def recommendation_must_be_valid(cls, v):
        valid_recommendations = ["Strong Match", "Potential Match", "Not Recommended"]
        if v not in valid_recommendations:
            raise ValueError(f'Recommendation must be one of: {", ".join(valid_recommendations)}')
        return v

class CandidateScreeningAgent:
    """Agent for screening job candidates using local LLMs"""
    
   

    def __init__(self, job_description: str):
        """
        Initialize the screening agent using Google Gemini.

        Args:
            job_description: The job description to screen candidates against
        """
        self.job_description = job_description

        # Use Google Gemini (gemini-pro)
        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.0-flash",
            api_key=API_KEY,
            temperature=0.1
        )

        self.output_parser = PydanticOutputParser(pydantic_object=CandidateAssessment)

    
  
    def load_resume(self, file_path: str) -> str:
        """
        Load and extract text from a resume file
        
        Args:
            file_path: Path to the resume file (PDF, DOCX, or TXT)
            
        Returns:
            Extracted text from the resume
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
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
    
    
    
    
    

    def create_assessment_prompt(self, resume_text: str) -> PromptTemplate:
        """Create the prompt template for candidate assessment"""
    
        format_instructions = self.output_parser.get_format_instructions()
        template="""
You are an expert HR recruiter. Analyze the candidate's resume against the job description below.

Return ONLY a valid JSON object that strictly follows the schema. Do NOT include explanations or formatting like code blocks. The response MUST be a plain JSON object.

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

Schema:
{{
  "candidate_name": "string",
  "skills_match_score": integer (1-10),
  "experience_relevance_score": integer (1-10),
  "education_match_score": integer (1-10),
  "overall_fit_score": integer (1-10),
  "strengths": [string],
  "weaknesses": [string],
  "recommendation": "Strong Match" | "Potential Match" | "Not Recommended",
  "candidate_email": "string"
}}
"""

       
    
        return PromptTemplate(
        input_variables=["job_description", "resume_text"],
        partial_variables={"format_instructions": format_instructions},
        template=template
    )
        
   
    def screen_candidate(self, resume_path: str) -> CandidateAssessment:
        """
        Screen a candidate's resume against the job description
        
        Args:
            resume_path: Path to the candidate's resume file
            
        Returns:
            Structured assessment of the candidate
        """
        # Load and extract text from resume
        resume_text = self.load_resume(resume_path)
        
        # Create prompt for assessment
        prompt = self.create_assessment_prompt(resume_text)
        
        #build new chain
        
        chain= prompt | self.llm | self.output_parser 
        result = None
        try:
            result =chain.invoke({
                "job_description":self.job_description,
                "resume_text":resume_text
            })
            return result
        except Exception as e:
            print(e)
            print(f"\nError Parsing output:\n{e}\n")
            print("=== RAW LLM OUTPUT START ===")
            print(result)
            print("=== RAW LLM OUTPUT END ===")
            return None
        
    def batch_screen_candidates(self, resume_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Screen multiple candidates and return assessments
        
        Args:
            resume_paths: List of paths to candidate resume files
            
        Returns:
            List of candidate assessments
        """
        results = []
        
        for path in resume_paths:
            candidate_name = os.path.basename(path).split('.')[0]
            print(f"Screening candidate: {candidate_name}")
            
            try:
                assessment = self.screen_candidate(path)
                
                if isinstance(assessment, CandidateAssessment):
                    results.append(assessment.dict())
                else:
                    # Handle case where parsing failed
                    results.append({
                        "candidate_name": candidate_name,
                        "raw_assessment": assessment,
                        "error": "Failed to parse structured output"
                    })
            except Exception as e:
                print(f"Error screening candidate {candidate_name}: {e}")
                traceback.print_exc()
                results.append({
                    "candidate_name": candidate_name,
                    "error": str(e)
                })
        
        return results
    
 

    def generate_report(self, assessments: List[Dict[str, Any]], output_path: str = "candidate_assessments.csv"):
        """
    Generate a CSV report from candidate assessments, including KMeans-based PASS/FAIL status.
    
    Args:
        assessments: List of candidate assessments
         output_path: Path to save the CSV report
        """
        df = pd.DataFrame(assessments)

    # Apply KMeans clustering if 'overall_fit_score' exists
        if 'overall_fit_score' in df.columns:
          scores = df['overall_fit_score'].values.reshape(-1, 1)
          kmeans = KMeans(n_clusters=3, random_state=42)
          df['cluster'] = kmeans.fit_predict(scores)

        # Identify top-performing cluster based on centroid value
          centroids = kmeans.cluster_centers_.flatten()
          top_cluster = np.argmax(centroids)  # cluster with highest average score

        # Assign PASS/FAIL based on cluster membership
          df['status'] = df['cluster'].apply(lambda x: 'PASS' if x == top_cluster else 'FAIL')

    # Save report to CSV
        df.to_csv(output_path, index=False)
        print(f"Report generated and saved to {output_path}")

               
        if 'overall_fit_score' in df.columns:
           top_candidates = df.sort_values(by='overall_fit_score', ascending=False).head(5)
           print("\nTop 5 Candidates:")
           display_cols = ['candidate_name', 'overall_fit_score', 'recommendation', 'candidate_email']
           if 'status' in df.columns:
             display_cols.append('status')
    
           print(top_candidates[display_cols])

           receiver = []
           if (top_candidates['overall_fit_score'] > 6).any():
            high_scorers = top_candidates[top_candidates['overall_fit_score'] > 6]
            print("Candidates with score > 6:")
            print(high_scorers[['candidate_name', 'candidate_email']])
        
        # Extract list of emails from high_scorers
            receiver = high_scorers['candidate_email'].tolist()

           if len(receiver) > 0:
            message = "Congratulations! You have been shortlisted based on your profile."
            # print(job_description)
            # receiver=["abhijeetsrivastava2189@gmail.com"]
            #job desc current stage next stage
            current_stage="Screening Phase"
            next_stage="Interview Round"
            receiver.append("abhijeetsrivastava2189@gmail.com")
            receiver.append("Aurjobsa@gmail.com")
            
            send_bulk_email(receiver,job_description,current_stage,next_stage)
 
               
             

            
            #   print(top_candidates[display_cols])
        # print("---------")
        # print(top_candidates['overall_fit_score'].apply(type))
    # def generate_report(self, assessments: List[Dict[str, Any]], output_path: str = "candidate_assessments.csv"):
    #     """
    #     Generate a CSV report from candidate assessments
        
    #     Args:
    #         assessments: List of candidate assessments
    #         output_path: Path to save the CSV report
    #     """
    #     df = pd.DataFrame(assessments)
    #     df.to_csv(output_path, index=False)
    #     print(f"Report generated and saved to {output_path}")
        
    #     # Sort and display top candidates
    #     if 'overall_fit_score' in df.columns:
    #         top_candidates = df.sort_values(by='overall_fit_score', ascending=False).head(5)
    #         print("\nTop 5 Candidates:")
    #         print(top_candidates[['candidate_name', 'overall_fit_score', 'recommendation']])


# Example usage
if __name__ == "__main__":
    # Example job description
    job_description = """
    Senior Software Engineer - AI/ML Team
    
    Requirements:
    - 5+ years of experience in software engineering
    - Proficiency in Python and experience with machine learning frameworks (TensorFlow, PyTorch)
    - Experience with large language models and natural language processing
    - Strong background in data structures and algorithms
    - Bachelor's degree in Computer Science or related field (Master's preferred)
    - Excellent communication and teamwork skills
    
    Responsibilities:
    - Develop and maintain AI/ML models and pipelines
    - Collaborate with cross-functional teams to implement AI features
    - Optimize models for production environments
    - Mentor junior engineers
    """
    
    # # Initialize the screening agent
    # agent = CandidateScreeningAgent(
    #     job_description=job_description,
    #     llm_type="ollama",  # Use "llamacpp" if you have models downloaded locally
    #     model_name="llama2"  # or another model you have available
    # )
    
    # Initialize the screening agent with Gemini
    agent = CandidateScreeningAgent(
         job_description=job_description
)
    
    # Example of screening a single candidate
    # assessment = agent.screen_candidate("path/to/resume.pdf")
    # print(assessment)
    
    # Example of batch screening
    resume_folder = "server/resumes"
    if os.path.exists(resume_folder):
        resume_paths = [os.path.join(resume_folder, f) for f in os.listdir(resume_folder) 
                       if f.endswith(('.pdf', '.docx', '.txt'))]
        
        assessments = agent.batch_screen_candidates(resume_paths)
        agent.generate_report(assessments)
    else:
        print(f"Resume folder {resume_folder} not found. Please create it and add resumes.")


# Utility Function: Add a simple resume parser for extraction of key information
def extract_key_info_from_resume(resume_text: str) -> Dict[str, Any]:
    """
    Extract key information from a resume
    
    Args:
        resume_text: Text content of the resume
        
    Returns:
        Dictionary containing extracted information
    """
    info = {
        "name": None,
        "email": None,
        "phone": None,
        "education": [],
        "experience": [],
        "skills": []
    }
    
    # Extract email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_matches = re.findall(email_pattern, resume_text)
    if email_matches:
        info["email"] = email_matches[0]
    
    # Extract phone
    phone_pattern = r'\b(?:\+\d{1,3}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'
    phone_matches = re.findall(phone_pattern, resume_text)
    if phone_matches:
        info["phone"] = phone_matches[0]
    
    # Education section detection
    education_section = re.search(r'(?i)education.*?(?=experience|skills|$)', resume_text, re.DOTALL)
    if education_section:
        edu_text = education_section.group(0)
        # Look for degree patterns
        degree_patterns = [
            r'\b(?:B\.?S\.?|Bachelor of Science|Bachelor\'s)\b.*?(?:\d{4}|\d{2})',
            r'\b(?:M\.?S\.?|Master of Science|Master\'s)\b.*?(?:\d{4}|\d{2})',
            r'\b(?:Ph\.?D\.?|Doctor of Philosophy|Doctorate)\b.*?(?:\d{4}|\d{2})'
        ]
        
        for pattern in degree_patterns:
            matches = re.findall(pattern, edu_text)
            info["education"].extend(matches)
    
    # Skills section detection (simple approach)
    skills_section = re.search(r'(?i)skills.*?(?=experience|education|$)', resume_text, re.DOTALL)
    if skills_section:
        skills_text = skills_section.group(0)
        # Extract words that might be skills
        potential_skills = re.findall(r'\b[A-Za-z+#\.]+\b', skills_text)
        # Filter out common words
        common_words = {'skills', 'and', 'the', 'with', 'in', 'of', 'a', 'to', 'for'}
        info["skills"] = [s for s in potential_skills if len(s) > 2 and s.lower() not in common_words]
    
    return info