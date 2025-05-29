import requests
import os

call_payload = {
    "From": "8887596182",
    "To": "9721558140",
    "CallerId": "80-458-83404",
    "Priority": "high",
    "TimeLimit": "600",  # 10 minutes
    "TimeOut": "30"
}

SID = "aurjobs1"  # your SID
API_URL = f"https://api.exotel.com/v1/Accounts/{SID}/Calls/connect.json"

# Auth using API key and token
API_KEY = "288d3c094bfce1a16e0c28caceec158e2d03d61bf4990093"
API_TOKEN = "e0406e45315a7509292f2889baed3fb119146e74ec94808c"

response = requests.post(
    API_URL,
    auth=(API_KEY, API_TOKEN),
    data=call_payload
)

print("Status Code:", response.status_code)
print("Response:", response.json() if response.headers.get('content-type') == 'application/json' else response.text)


# import requests
# import os
# call_payload = {
#             "From":"8887596182",
#             "To": "9721558140",
#             "CallerId": "80-458-83404",
#             # "Url": f"{os.getenv('WEBHOOK_BASE_URL')}/ivr/{candidate['id']}",
#             "Priority": "high",
#             "TimeLimit": "600",  # 10 minutes
#             "TimeOut": "30"
#         }
# auth = (os.getenv("EXOTEL_SID"), os.getenv("EXOTEL_TOKEN"))
# response = requests.post(
#             f"https://288d3c094bfce1a16e0c28caceec158e2d03d61bf4990093:e0406e45315a7509292f2889baed3fb119146e74ec94808capi.exotel.com/v1/Accounts/aurjobs1/Calls/connect'",
#             # f"https://api.exotel.com/v1/Accounts/{os.getenv('EXOTEL_SID')}/Calls/connect",
#             auth=auth,
#             data=call_payload
#         ) 
        
# print(response.json())