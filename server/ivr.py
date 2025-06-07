from fastapi import FastAPI, Request, Form
from fastapi.responses import PlainTextResponse
import json
import os
import datetime

app = FastAPI()

# Handle Exotel Passthru applet
@app.get("/ivr/{candidate_id}")
async def passthru_handler(candidate_id: int, request: Request):
    print("jai mata di")
    params = dict(request.query_params)
    
    # Optional: Save passthru details for logging or CRM sync
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"passthru_log_{candidate_id}_{now}.txt"
    with open(filename, "w") as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    
    # return PlainTextResponse("OK", status_code=200)
    return PlainTextResponse(f"""
        <Response>
            <Say>Hey you are Abhijeet.</Say>
            <Hangup />
        </Response>
        """, media_type="application/xml")

# Handle IVR question and recording response
@app.post("/ivr/{candidate_id}")
async def ivr_handler(candidate_id: int, request: Request):
    params = dict(await request.form())
    question_index = int(params.get("question_index", 0))

    # Load questions for this candidate
    question_file = f"temp_questions_{candidate_id}.json"
    if not os.path.exists(question_file):
        return PlainTextResponse(f"""
        <Response>
            <Say>Sorry, we could not find your interview questions. Please contact support.</Say>
            <Hangup />
        </Response>
        """, media_type="application/xml")

    with open(question_file) as f:
        question_data = json.load(f)
    questions = question_data["questions"]

    # Save previous recording (if any)
    if "RecordingUrl" in params and question_index > 0:
        previous_question = questions[question_index - 1]
        recording_url = params.get("RecordingUrl")

        response_file = f"response_{candidate_id}_q{question_index - 1}.json"
        with open(response_file, "w") as f:
            json.dump({
                "question": previous_question,
                "audio_url": recording_url,
                "question_index": question_index - 1
            }, f)
        print(f"Saved response to: {response_file}")

    # End interview if all questions are done
    if question_index >= len(questions):
        return PlainTextResponse(f"""
        <Response>
            <Say>Thank you. Your interview is complete.</Say>
            <Hangup />
        </Response>
        """, media_type="application/xml")

    # Ask current question and record response
    current_question = questions[question_index]
    return PlainTextResponse(f"""
    <Response>
        <Say>{current_question}</Say>
        <Record
            action="/ivr/{candidate_id}?question_index={question_index + 1}"
            method="POST"
            maxLength="120"
            timeout="5"
            playBeep="true"
        />
    </Response>
    """, media_type="application/xml")




# from fastapi import FastAPI, Request
# from fastapi.responses import PlainTextResponse
# import json
# import os

# app = FastAPI()

# @app.post("/ivr/{candidate_id}")
# async def ivr_handler(candidate_id: int, request: Request):
#     params = dict(await request.form())
#     question_index = int(params.get("question_index", 0))

#     # Load questions file safely
#     question_file = f"temp_questions_{candidate_id}.json"
#     if not os.path.exists(question_file):
#         return PlainTextResponse(f"""
#         <Response>
#             <Say>Sorry, we could not find your interview questions. Please contact support.</Say>
#             <Hangup />
#         </Response>
#         """, media_type="application/xml")

#     with open(question_file) as f:
#         question_data = json.load(f)
#     questions = question_data["questions"]

#     # Save previous recording (if available)
#     if "RecordingUrl" in params and question_index > 0:
#         previous_question = questions[question_index - 1]
#         recording_url = params.get("RecordingUrl")

#         response_file = f"response_{candidate_id}_q{question_index - 1}.json"
#         with open(response_file, "w") as f:
#             json.dump({
#                 "question": previous_question,
#                 "audio_url": recording_url,
#                 "question_index": question_index - 1
#             }, f)
#         print(f"Saved response to: {response_file}")

#     # End the interview if all questions are done
#     if question_index >= len(questions):
#         return PlainTextResponse(f"""
#         <Response>
#             <Say>Thank you. Your interview is complete.</Say>
#             <Hangup />
#         </Response>
#         """, media_type="application/xml")

#     # Ask current question and record answer
#     current_question = questions[question_index]
#     return PlainTextResponse(f"""
#     <Response>
#         <Say>{current_question}</Say>
#         <Record
#             action="/ivr/{candidate_id}?question_index={question_index + 1}"
#             method="POST"
#             maxLength="120"
#             timeout="5"
#             playBeep="true"
#         />
#     </Response>
#     """, media_type="application/xml")
