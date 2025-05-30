# ai_agent.py
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
# from langchain.llms import Ollama  # Free local LLM
# from langchain_community.llms import Ollama
# from langchain.llms import Ollama
# from langchain_ollama import OllamaLLM
from langchain_community.chat_models import ChatOllama
from langchain.chains import LLMChain
from langchain.agents.output_parsers import ReActSingleInputOutputParser

from langchain_google_genai import ChatGoogleGenerativeAI
import warnings
warnings.filterwarnings("ignore")

#celery 
#candidate response time kam hona chahiye particular response time
#limit the response for questions
from langchain.schema import AgentAction, AgentFinish
from langchain.memory import ConversationBufferMemory
import requests
import whisper
import json
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()
import os
print(os.getenv("GOOGLE_API_KEY"))
print(os.getenv("EXOTEL_API_KEY"))
print(os.getenv("EXOTEL_API_TOKEN"))
class PreScreeningAgent:
    def __init__(self):
        # Initialize free tools
        # self.llm = ChatOllama(model="mistral")  # Free local Mistral
        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.0-flash",
            
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1
        )
        self.whisper_model = whisper.load_model("base")  # Free transcription
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        
        # Setup tools
        self.tools = [
            Tool(
                name="pre_screen_candidates",
                func=self.run_pre_screening,
                description="Conduct AI voice pre-screening for job candidates. Input: {'candidates': [...], 'job_description': '...'}"
            ),
            Tool(
                name="generate_questions",
                func=self.generate_screening_questions,
                description="Generate job-specific screening questions. Input: job_description"
            ),
            Tool(
                name="evaluate_candidate",
                func=self.evaluate_single_candidate,
                description="Evaluate a single candidate's responses. Input: {'candidate': {...}, 'responses': [...]}"
            )
        ]
        
        # Create agent
        self.agent = self._create_agent()
    
    def _create_agent(self):
        prompt_template = """
        You are an AI recruitment assistant specializing in candidate pre-screening.
        
        Available tools: {tools}
        Tool names: {tool_names}
        
        Current conversation:
        {chat_history}
        
        Human: {input}
        
        Think step by step:
        Thought: I need to understand what the human wants
        Action: [tool_name]
        Action Input: [input_to_tool]
        Observation: [result_from_tool]
        ... (repeat Thought/Action/Action Input/Observation as needed)
        Thought: I now know the final answer
        Final Answer: [final_response]
        
        {agent_scratchpad}
        """
        
        class CustomPromptTemplate(StringPromptTemplate):
            template: str
            tools: List[Tool]
            
            def format(self, **kwargs) -> str:
                kwargs['tools'] = '\n'.join([f"{tool.name}: {tool.description}" for tool in self.tools])
                kwargs['tool_names'] = ', '.join([tool.name for tool in self.tools])
                return self.template.format(**kwargs)
        
        prompt = CustomPromptTemplate(
            template=prompt_template,
            tools=self.tools,
            input_variables=["input", "chat_history", "agent_scratchpad"]
        )
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        
        return LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=ReActSingleInputOutputParser(),
            prompt=prompt,
            stop=["\nObservation:"],
            allowed_tools=[tool.name for tool in self.tools]
        )
    
    def generate_screening_questions(self, job_description: str) -> List[str]:
        """Generate job-specific screening questions using free local LLM"""
        prompt = f"""
        Generate exactly 3 specific screening questions for this job:
        
        Job Description: {job_description}
        
        Questions should be:
        1. Technical/skill-based
        2. Experience-focused
        3. Scenario-based
        
        Format: Return only questions, one per line.
        """
        # print(prompt)
        response = self.llm.invoke(prompt)
        # print(response)
        # questions = [q.strip() for q in response.split('\n') if q.strip()]
        questions = [q.strip() for q in response.content.split('\n') if q.strip()]

        # print(questions)
        return questions[:3]  # Ensure exactly 3 questions
    
    
    # def trigger_exotel_call(self, candidate: Dict, questions: List[str]) -> Dict:
        """Trigger Exotel call with dynamic IVR"""
        # Store questions in temporary storage for IVR access
        question_data = {
            "candidate_id": candidate["id"],
            "questions": questions,
            "timestamp": time.time()
        }
        
        # Save to file/database for IVR webhook access
        with open(f"temp_questions_{candidate['id']}.json", 'w') as f:
            json.dump(question_data, f)
        
        call_payload = {
            "From":"8887596182",
            "To": "9721558140",
            "CallerId": "80-458-83404",
            # "Url": f"{os.getenv('WEBHOOK_BASE_URL')}/ivr/{candidate['id']}",
            "Priority": "high",
            "TimeLimit": "600",  # 10 minutes
            "TimeOut": "30"
        }
        
        auth = (os.getenv("EXOTEL_SID"), os.getenv("EXOTEL_TOKEN"))
        response = requests.post(
            f"https://288d3c094bfce1a16e0c28caceec158e2d03d61bf4990093:e0406e45315a7509292f2889baed3fb119146e74ec94808capi.exotel.com/v1/Accounts/aurjobs1/Calls/connect'",
            # f"https://api.exotel.com/v1/Accounts/{os.getenv('EXOTEL_SID')}/Calls/connect",
            auth=auth,
            data=call_payload
        ) 
        
        return response.json()
    
    # 
    def trigger_exotel_call(self, candidate: Dict, questions: List[str]) -> Dict:
        """Trigger Exotel call with dynamic IVR"""
    

     # Store questions in temporary storage for IVR access
        question_data = {
        "candidate_id": candidate["id"],
        "questions": questions,
        "timestamp": time.time()
         }
        # print("here are the questions ",questions)
     # Save to file/database for IVR webhook access
        with open(f"temp_questions_{candidate['id']}.json", 'w') as f:
         json.dump(question_data, f)

     # Prepare call payload
        call_payload = {
        "From": "8887596182",
        # "To": "9721558140",  # Use candidate's phone if available
        "To":candidate["phone"],
        "CallerId": "80-458-83404",
        # "Url": f"{os.getenv('WEBHOOK_BASE_URL')}/ivr/{candidate['id']}",
        "Priority": "high",
        "TimeLimit": "600",  # 10 minutes
        "TimeOut": "30"
            }

     # API credentials
        
        api_key=os.getenv("EXOTEL_API_KEY")
        api_token=os.getenv("EXOTEL_API_TOKEN")
        sid = "aurjobs1"
        # base_url = "https://api.exotel.com/v1"
        print(candidate["phone"],"-----")
        API_URL = f"https://api.exotel.com/v1/Accounts/{sid}/Calls/connect.json"
        

        try:
         response = requests.post(
            API_URL,
            auth=(api_key, api_token),
            data=call_payload
         )
        #  print("00000000")
         print(response)
         response.raise_for_status()
         return response.json()
        except requests.RequestException as e:
         return {
            "error": "Failed to initiate call",
            "details": str(e),
            "response": response.text if 'response' in locals() else None
        }

    def transcribe_audio(self, audio_url: str) -> str:
        """Transcribe audio using free Whisper"""
        print("0000000000")
      
        try:
         
         if not audio_url.startswith("http"):
            raise ValueError(f"Invalid audio URL: {audio_url}")
        
         audio_response = requests.get(audio_url)
         if audio_response.status_code != 200:
            raise ValueError(f"Failed to download audio: {audio_response.status_code}")
        
         temp_file = f"temp_audio_{int(time.time())}.mp3"
         with open(temp_file, "wb") as f:
            f.write(audio_response.content)
        
         # Transcribe
         result = self.whisper_model.transcribe(temp_file)
         print(result)
         # Cleanup
         os.remove(temp_file)
        
         return result["text"].strip()
        except Exception as e:
         print(f"Transcription error: {e}")
        return ""

    def evaluate_answer(self, question: str, answer: str) -> float:
        """Evaluate answer using free local LLM"""
        if not answer.strip():
            return 0.0
        
        prompt = f"""
        Evaluate this interview answer on a scale of 0-10:
        
        Question: {question}
        Answer: {answer}
        
        Consider:
        - Relevance to question
        - Technical accuracy
        - Communication clarity
        - Experience depth
        
        Return only a number between 0-10:
        """
        
        response = self.llm.invoke(prompt)
        
        try:
            # Extract number from response
            score_str = ''.join(filter(str.isdigit, response.split()[0]))
            score = float(score_str) if score_str else 0.0
            return min(max(score, 0.0), 10.0)  # Clamp between 0-10
        except:
            return 0.0
    
    def wait_for_responses(self, candidate_id: int, num_questions: int, timeout: int = 30):
        """Wait for webhook responses with timeout (15 minutes)"""
        start_time = time.time()
        responses = []
        
        while len(responses) < num_questions and (time.time() - start_time) < timeout:
            # Check for response files from webhook
            for i in range(num_questions):
                response_file = f"responses_{candidate_id}_q{i+1}.json"
                if os.path.exists(response_file):
                    with open(response_file, 'r') as f:
                        response_data = json.load(f)
                    responses.append(response_data)
                    os.remove(response_file)  # Cleanup
            
            time.sleep(2)  # Check every 2 seconds
        
        return responses
    
    def run_pre_screening(self, input_data: str) -> Dict:
        """Main pre-screening function"""
        print("ja ho")
        try:
            # Parse input
            data = json.loads(input_data) if isinstance(input_data, str) else input_data
            candidates = data.get("candidates", [])
            # print(candidates)
            job_description = data.get("job_description", "")
            # print(job_description)
            
            if not candidates or not job_description:
                return {"error": "Missing candidates or job description"}
            
            # Generate questions
            print("------------")
            questions = self.generate_screening_questions(job_description)
            # print("+++++++++++++")
            # print(questions)
            
            results = []
            
            for candidate in candidates:
                print(f"Starting pre-screening for {candidate.get('name', 'Unknown')}")
                
                # Trigger call
                call_result = self.trigger_exotel_call(candidate, questions)
                
                if "error" in call_result:
                    results.append({
                        "candidate_id": candidate.get("id"),
                        "name": candidate.get("name"),
                        "status": "call_failed",
                        "error": call_result["error"],
                        "score": 0.0
                    })
                    continue
                
                # Wait for responses
                
                responses = self.wait_for_responses(candidate.get("id"), len(questions))
                
                if not responses:
                    results.append({
                        "candidate_id": candidate.get("id"),
                        "name": candidate.get("name"),
                        "status": "no_response",
                        "score": 0.0
                    })
                    continue
                
                # Process responses
                scores = []
                transcripts = []
                
                for i, response in enumerate(responses):
                    if i < len(questions):
                        audio_url = response.get("audio_url", "")
                        if audio_url:
                            transcript = self.transcribe_audio(audio_url)
                            score = self.evaluate_answer(questions[i], transcript)
                            scores.append(score)
                            transcripts.append({
                                "question": questions[i],
                                "answer": transcript,
                                "score": score
                            })
                
                avg_score = sum(scores) / len(scores) if scores else 0.0
                
                results.append({
                    "candidate_id": candidate.get("id"),
                    "name": candidate.get("name"),
                    "phone": candidate.get("phone"),
                    "status": "completed",
                    "overall_score": round(avg_score, 2),
                    "individual_scores": scores,
                    "responses": transcripts,
                    "qualified": avg_score >= 6.0  # Threshold for next round
                })
            
            # Sort by score (highest first)
            qualified_candidates = sorted(
                [r for r in results if r.get("qualified", False)],
                key=lambda x: x.get("overall_score", 0),
                reverse=True
            )
            
            return {
                "status": "success",
                "total_candidates": len(candidates),
                "completed_screenings": len([r for r in results if r["status"] == "completed"]),
                "qualified_count": len(qualified_candidates),
                "qualified_candidates": qualified_candidates,
                "all_results": results
            }
            
        except Exception as e:
            return {"error": f"Pre-screening  failed: {str(e)}"}
    
    def evaluate_single_candidate(self, input_data: str) -> Dict:
        """Evaluate a single candidate's performance"""
        try:
            data = json.loads(input_data) if isinstance(input_data, str) else input_data
            candidate = data.get("candidate", {})
            responses = data.get("responses", [])
            
            if not responses:
                return {"error": "No responses to evaluate"}
            
            scores = [r.get("score", 0) for r in responses]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            
            # Detailed analysis
            strengths = []
            weaknesses = []
            
            for response in responses:
                if response.get("score", 0) >= 7:
                    strengths.append(response.get("question", ""))
                elif response.get("score", 0) <= 4:
                    weaknesses.append(response.get("question", ""))
            
            recommendation = "PROCEED" if avg_score >= 6.0 else "REJECT"
            
            return {
                "candidate_name": candidate.get("name"),
                "overall_score": round(avg_score, 2),
                "recommendation": recommendation,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "detailed_scores": scores
            }
            
        except Exception as e:
            return {"error": f"Evaluation failed: {str(e)}"}

# Main execution function for external calls
def create_prescreening_agent():
    """Factory function to create agent instance"""
    return PreScreeningAgent()

# Example usage
if __name__ == "__main__":
    agent = create_prescreening_agent()
    
    # Test data
    test_input = {
        "candidates": [
            {"id": 1, "name": "John Doe", "phone": "9721558140"},
           
        ],
        "job_description": "Senior Python Developer with Django and PostgreSQL experience"
    }
    
    result = agent.run_pre_screening(json.dumps(test_input))
    print(json.dumps(result, indent=2))