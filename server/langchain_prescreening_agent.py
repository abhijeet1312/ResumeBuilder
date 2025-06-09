# ai_agent.py
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
#
from langchain_community.chat_models import ChatOllama
import traceback
from langchain.chains import LLMChain
from langchain.agents.output_parsers import ReActSingleInputOutputParser

from langchain_google_genai import ChatGoogleGenerativeAI
import warnings
from requests.auth import HTTPBasicAuth
warnings.filterwarnings("ignore")
import os
from twilio.rest import Client

#celery c
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
        Generate exactly 2 specific screening questions for this job:
        
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
    
    
   


    def trigger_exotel_call(self, candidate: Dict, questions: List[str]) -> Dict:
        """Trigger Twilio call with dynamic IVR based on candidate and questions."""

        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        client = Client(account_sid, auth_token)

        candidate_id = candidate.get("id")
    # candidate_phone = candidate.get("phone")  # Ensure candidate dict has 'phone'
        candidate_phone ="+918887596182"  # Ensure candidate dict has 'phone'

        if not candidate_id or not candidate_phone:
         return {"success": False, "error": "Candidate ID or phone missing"}

    # Save questions to temp file for IVR to read
        question_data = {
        "candidate_id": candidate_id,
        "questions": questions,
        "timestamp": time.time()
         }

        temp_file_path = os.path.join(os.path.dirname(__file__), f"temp_questions_{candidate_id}.json")
        with open(temp_file_path, 'w') as f:
         json.dump(question_data, f)

        try:
        # Trigger the outbound call via Twilio
          call = client.calls.create(
            from_="+17178825763",  # your verified Twilio number
            to=candidate_phone,   # dynamic recipient
            url=f"https://ab39-2405-201-600e-f151-537-6f75-9239-3b9f.ngrok-free.app/voice/{candidate['id']}"  # must match your webhook
           )
          print(call)
          return {"success": True, "call_sid": call.sid}
        except Exception as e:
          return {"success": False, "error": str(e)}
      
      
    

    def transcribe_audio(self, audio_url: str) -> str:
      print("inside transcribe audio")
      temp_file = None
      try:
        if not audio_url.startswith("http"):
            raise ValueError(f"Invalid audio URL: {audio_url}")

        audio_response = requests.get(
            audio_url,
            auth=HTTPBasicAuth(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
        )
        print("jai mata di ")
        
        if audio_response.status_code != 200:
            raise ValueError(f"Failed to download audio: {audio_response.status_code}")

        # Use absolute path and unique filename
        import uuid
        script_dir = os.path.dirname(os.path.abspath(__file__))
        temp_filename = f"temp_audio_{uuid.uuid4().hex[:8]}.mp3"
        temp_file = os.path.join(script_dir,temp_filename)
        
        print(f"📁 Saving audio to: {temp_file}")
        
        with open(temp_file, "wb") as f:
            f.write(audio_response.content)
            f.flush() #ensure data is written to disk
            os.fsync(f.fileno()) # force write to disk
        
        # Verify file was written
        if not os.path.exists(temp_file):
            raise FileNotFoundError(f"Failed to create temporary file: {temp_file}")
            
        file_size = os.path.getsize(temp_file)
        print(f"📊 File size: {file_size} bytes")
        
        if file_size < 100:  # Reduced minimum size threshold
            raise ValueError(f"Audio file too small ({file_size} bytes) - likely corrupted or empty")

        time.sleep(5)
        print(f"🎵 Transcribing: {temp_file}")
        
        # Triple-check file exists before transcription with detailed debug info
        # print(f'Current working directory: {os.getcwd()}')
        print(f'Script directory: {script_dir}')
        print(f'Temp file path: {temp_file}')
        print(f'Temp file absolute path: {os.path.abspath(temp_file)}')
        print(f'File exists: {os.path.exists(temp_file)}')
        
        # Double-check file exists before transcription
        # print('ospath',os.path)
        if not os.path.exists(temp_file):
            print('file not found in os path')
            raise FileNotFoundError(f"Temporary file disappeared: {temp_file}")
        
        # Ensure whisper_model is properly initialized
        if not hasattr(self, 'whisper_model') or self.whisper_model is None:
            raise AttributeError("Whisper model not initialized")
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
           
        abs_temp_file = os.path.abspath(temp_file)
        print(f"Using absolute path for transcription: {abs_temp_file}")
        
        
        try:
            print("🎤 Attempting direct audio loading with librosa...")
            import librosa
            import numpy as np
                
                # Save to temp file first
            direct_temp_file = os.path.join(os.getcwd(), f"direct_{uuid.uuid4().hex[:6]}.mp3")
                
            with open(direct_temp_file, "wb") as f:
                 f.write(audio_response.content)
                 f.flush()
                 os.fsync(f.fileno())
                
                # Load audio with librosa
            audio_data, sr = librosa.load(direct_temp_file, sr=16000)  # Whisper expects 16kHz
                
                # Clean up temp file immediately
            if os.path.exists(direct_temp_file):
                    os.remove(direct_temp_file)
                
                # Pass numpy array directly to Whisper
            result = self.whisper_model.transcribe(audio_data)
            print("✅ Direct audio loading successful!")
                
        except Exception as direct_audio_error:
             print(f"❌ Direct audio loading failed: {direct_audio_error}")
        
        
        
       
        print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")
        
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"🗑️ Cleaned up temporary file: {temp_file}")
            
        return result["text"].strip()

      except Exception as e:
        print(f"Transcription error: {e}")
        print(f"Error type: {type(e).__name__}")
        
        
        # Additional debugging info
        if temp_file:
            print(f"Debug - temp_file variable: {temp_file}")
            print(f"Debug - temp_file exists: {os.path.exists(temp_file)}")
            print(f"Debug - temp_file absolute: {os.path.abspath(temp_file)}")
        
        # Clean up temp file if it exists
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                print(f"🗑️ Cleaned up temporary file after error: {temp_file}")
            except Exception as cleanup_error:
                print(f"Failed to cleanup temp file: {cleanup_error}")
        
        return ""

    
    def evaluate_answer(self, question: str, answer: str) -> float:
        """Evaluate answer using free local LLM"""
        print("inside evaluate answer")
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
        print("response of question and answer",response)
        
        # Extract the actual content (the string '6', for example)
        if hasattr(response, "content"):
          answer_text = response.content
        else:
          answer_text = str(response)  # fallback
        
        
        
        try:
            # Extract number from response
            score_str = ''.join(filter(str.isdigit, answer_text.split()[0]))
            score = float(score_str) if score_str else 0.0
            return min(max(score, 0.0), 10.0)  # Clamp between 0-10
        except:
            return 0.0
    
 

   
    
    def wait_for_responses(self, candidate_id: int, num_questions: int, timeout: int = 300):
      """Wait for webhook responses with timeout (5 minutes)"""
      start_time = time.time()
      responses = []
      received = set()
    
     # Add current working directory info for debugging
      current_dir = os.getcwd()
      print(f"🔍 Looking for files in: {current_dir}")
      print(f"⏳ Waiting for {num_questions} response files for candidate {candidate_id}...")

      while len(responses) < num_questions and (time.time() - start_time) < timeout:
        # Show progress every 30 seconds
        elapsed = time.time() - start_time
        if int(elapsed) % 30 == 0 and elapsed > 0:
            print(f"⏰ Still waiting... {elapsed:.0f}s elapsed, {len(responses)}/{num_questions} responses received")
        
        # Check for all response files
        for i in range(1, num_questions + 1):
            if i in received:
                continue

            response_file = f"responses_{candidate_id}_q{i}.json"
            
            if os.path.exists(response_file):
                print(f"📥 Found: {response_file}")
                
                # Add a small delay to ensure file is fully written
                time.sleep(0.1)
                
                try:
                    with open(response_file, 'r') as f:
                        response_data = json.load(f)
                    
                    # Validate the response data
                    if not response_data.get('audio_url'):
                        print(f"⚠️ Invalid response data in {response_file}: missing audio_url")
                        continue
                    
                    responses.append(response_data)
                    received.add(i)
                    
                    print(f"✅ Processed: {response_file} (Question {i})")
                    
                    # Don't delete the file immediately - keep for debugging
                    # You can uncomment this line later:
                    os.remove(response_file)
                    
                except Exception as e:
                    print(f"❌ Error reading {response_file}: {e}")
                    import traceback
                    traceback.print_exc()

        time.sleep(2)  # Check every 2 seconds

     # Final summary
      if len(responses) < num_questions:
        print(f"⚠️ Timeout reached after {timeout}s. Only received {len(responses)} of {num_questions} responses.")
        
        # List any files that exist but weren't processed
        all_response_files = [f for f in os.listdir('.') if f.startswith(f'responses_{candidate_id}_')]
        if all_response_files:
            print(f"🔍 Found these response files: {all_response_files}")
      else:
        print(f"✅ Successfully received all {num_questions} responses for candidate {candidate_id}")

      return responses


# Also add this helper method to your PreScreeningAgent class
    def check_call_status(self, call_sid: str) -> dict:
       """Check the status of a Twilio call"""
       try:
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        client = Client(account_sid, auth_token)
        
        call = client.calls(call_sid).fetch()
        return {
            "status": call.status,
            "duration": call.duration,
            "start_time": call.start_time,
            "end_time": call.end_time
        }
       except Exception as e:
        return {"error": str(e)}
    
    def run_pre_screening(self, input_data: str) -> Dict:
      """Main pre-screening function with better error handling"""
      print("Starting pre-screening process...")
      try:
        # Parse input
        data = json.loads(input_data) if isinstance(input_data, str) else input_data
        candidates = data.get("candidates", [])
        job_description = data.get("job_description", "")
        
        if not candidates or not job_description:
            return {"error": "Missing candidates or job description"}
        
        # Generate questions
        print("Generating screening questions...")
        questions = self.generate_screening_questions(job_description)
        print(f"Generated {len(questions)} questions: {questions}")
        
        results = []
        
        for candidate in candidates:
            candidate_name = candidate.get('name', 'Unknown')
            candidate_id = candidate.get('id')
            
            print(f"\n{'='*50}")
            print(f"Starting pre-screening for {candidate_name} (ID: {candidate_id})")
            print(f"{'='*50}")
            
            # Trigger call
            print("🔄 Initiating call...")
            call_result = self.trigger_exotel_call(candidate, questions)
            
            if "error" in call_result:
                print(f"❌ Call failed: {call_result['error']}")
                results.append({
                    "candidate_id": candidate_id,
                    "name": candidate_name,
                    "status": "call_failed",
                    "error": call_result["error"],
                    "score": 0.0
                })
                continue
            
            call_sid = call_result.get("call_sid")
            print(f"📞 Call initiated successfully. SID: {call_sid}")
            
            # Wait a bit for call to connect
            print("⏳ Waiting for call to connect...")
            time.sleep(10)
            
            # Check call status
            if call_sid:
                call_status = self.check_call_status(call_sid)
                print(f"📊 Call status: {call_status}")
            
            # Wait for responses with longer timeout
            print(f"🎧 Waiting for {len(questions)} responses...")
            responses = self.wait_for_responses(candidate_id, len(questions), timeout=300)  # 5 minutes
            
            if not responses:
                print(f"❌ No responses received for {candidate_name}")
                results.append({
                    "candidate_id": candidate_id,
                    "name": candidate_name,
                    "status": "no_response",
                    "score": 0.0,
                    "call_sid": call_sid
                })
                continue
            
            print(f"✅ Received {len(responses)} responses, processing...")
            
            # Process responses
            scores = []
            transcripts = []
            
            for i, response in enumerate(responses):
                if i < len(questions):
                    audio_url = response.get("audio_url", "")
                    if audio_url:
                        print(f"🎵 Transcribing audio for question {i+1}...")
                        transcript = self.transcribe_audio(audio_url)
                        print(f"📝 Transcript: {transcript[:100]}...")
                        
                        score = self.evaluate_answer(questions[i], transcript)
                        print(f"📊 Score for question {i+1}: {score}/10")
                        
                        scores.append(score)
                        transcripts.append({
                            "question": questions[i],
                            "answer": transcript,
                            "score": score
                        })
            
            avg_score = sum(scores) / len(scores) if scores else 0.0
            qualified = avg_score >= 2.0
            
            print(f"🎯 Final Results for {candidate_name}:")
            print(f"   Average Score: {avg_score:.2f}/10")
            print(f"   Qualified: {'✅ YES' if qualified else '❌ NO'}")
            
            results.append({
                "candidate_id": candidate_id,
                "name": candidate_name,
                "phone": candidate.get("phone"),
                "status": "completed",
                "overall_score": round(avg_score, 2),
                "individual_scores": scores,
                "responses": transcripts,
                "qualified": qualified,
                "call_sid": call_sid
            })
        
        # Sort by score (highest first)
        qualified_candidates = sorted(
            [r for r in results if r.get("qualified", False)],
            key=lambda x: x.get("overall_score", 0),
            reverse=True
        )
        
        print(f"\n{'='*50}")
        print("FINAL SCREENING RESULTS")
        print(f"{'='*50}")
        print(f"Total Candidates: {len(candidates)}")
        print(f"Completed Screenings: {len([r for r in results if r['status'] == 'completed'])}")
        print(f"Qualified Candidates: {len(qualified_candidates)}")
        
        return {
            "status": "success",
            "total_candidates": len(candidates),
            "completed_screenings": len([r for r in results if r["status"] == "completed"]),
            "qualified_count": len(qualified_candidates),
            "qualified_candidates": qualified_candidates,
            "all_results": results
        }
        
      except Exception as e:
        print(f"❌ Pre-screening failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Pre-screening failed: {str(e)}"}
    
    
    
    
    
    
    
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
    
    
    
    
