# import smtplib
# import os 
# from dotenv import load_dotenv
# load_dotenv()
# # creates SMTP session
# s = smtplib.SMTP('smtp.gmail.com', 587)
# # start TLS for security
# s.starttls()
# # Authentication
# s.login("shivamsrivastava2189@gmail.com", os.getenv("Google_app_password"))
# # message to be sent
# message = "Message_you_need_to_send"
# receipents=["kanaksrivastava1970@gmail.com","abhijeetsrivastava2189@gmail.com","anilanita07@gmail.com"]
# # sending the mail
# s.sendmail("shivamsrivastava2189@gmail.com", receipents, message)
# # terminating the session
# s.quit()

import smtplib
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
#single chain celery check krna hai
def send_bulk_email(recipients, job_description, current_stage, next_stage):
    import smtplib
    import os
    from langchain_google_genai import ChatGoogleGenerativeAI

    sender_email = "shivamsrivastava2189@gmail.com"
    app_password = os.getenv("Google_app_password")
    API_KEY = os.getenv("GOOGLE_API_KEY")

    model = ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash",
        api_key=API_KEY,
        temperature=0.7
    )

    prompt = f"""
    You are an HR assistant. Craft a professional, polite, and encouraging email to a job applicant.
    Inform them that they have successfully qualified the current stage of the hiring process.

    Job Description:
    {job_description}

    Current Stage:
    {current_stage}

    Next Stage:
    {next_stage}

    Ensure the email includes:
    - A congratulatory tone
    - Reference to the job role and selection stage
    - What the next stage involves
    - Next steps or instructions
    - Encouragement to prepare

    Keep it reusable and concise.
    """

    response = model.invoke(prompt)
    message = response.content if hasattr(response, "content") else response

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as s:
            s.starttls()
            s.login(sender_email, app_password)
            s.sendmail(sender_email, recipients, message)
            print("Email sent successfully to:", recipients)
    except Exception as e:
        print("Error sending email:", e)


# def send_bulk_email(recipients, message,job_description,current_stage,next_stage):
#     """
#     Send an email to multiple recipients using Gmail SMTP.
    
#     Args:
#         recipients (list): List of recipient email addresses
#         message (str): The message to be sent
#     """
#     sender_email = "shivamsrivastava2189@gmail.com"
#     app_password = os.getenv("Google_app_password")
    
#     API_KEY = os.getenv("GOOGLE_API_KEY")
#     print(API_KEY)
#     print("hi")
#     #convert_system_message_to_human=True
#     #load the model
#     model = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash",
#                                api_key=API_KEY,temperature=0.7)
    

#     try:
#         # Create SMTP session and start TLS
#         with smtplib.SMTP('smtp.gmail.com', 587) as s:
#             s.starttls()
#             s.login(sender_email, app_password)
#             s.sendmail(sender_email, recipients, message)
#             print("Email sent successfully to:", recipients)
#     except Exception as e:
#         print("Error sending email:", e)


# emails = [
#     "kanaksrivastava1970@gmail.com",
#     "abhijeetsrivastava2189@gmail.com",
#     "anilanita07@gmail.com"
# ]

# message = "Hello, this is a test message from Python!"
# send_bulk_email(emails, message)
