import os
import json
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# GitHub Secrets se token.json generate karna
token_data = os.environ.get("TOKEN_JSON")
if token_data:
    with open("token.json", "w") as f:
        f.write(token_data)

# Authenticate using token.json
if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json")
    youtube = build("youtube", "v3", credentials=creds)
else:
    raise FileNotFoundError("TOKEN_JSON file nahi mili!")

# Upload Video
request = youtube.videos().insert(
    part="snippet,status",
    body={
        "snippet": {
            "title": "Mind Blowing Daily Facts #Shorts",
            "description": script_text + "\n\n#shorts #viral #facts",
            "tags": ["shorts", "facts", "viral"],
            "categoryId": "27"
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": False
        }
    },
    media_body=MediaFileUpload("automated_daily_short.mp4")
)
response = request.execute()
print("Upload Successful! Video ID:", response.get('id'))
