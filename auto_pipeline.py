import os
import json
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# 1. Secret se token.json restore karein
token_data = os.environ.get("TOKEN_JSON")
if token_data:
    with open("token.json", "w") as f:
        f.write(token_data)

# 2. Agar aapke paas video generation ka koi script/function hai to usay yahan call karein:
# Example: os.system("python create_video.py")
# Ya agar aap script se hi video banate hain to wo function yahan run karein.

# 3. Video File Check Karein
video_file = "automated_daily_short.mp4"
if not os.path.exists(video_file):
    # Dummy video file check (Agar file generate nahi hui to error raise karein)
    raise FileNotFoundError(f"Video file '{video_file}' nahi mili! Pehle video generate karein.")

# 4. YouTube Upload
if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json")
    youtube = build("youtube", "v3", credentials=creds)
else:
    raise FileNotFoundError("TOKEN_JSON Secret missing hai!")

request = youtube.videos().insert(
    part="snippet,status",
    body={
        "snippet": {
            "title": "Mind Blowing Daily Facts #Shorts",
            "description": "Daily automated facts video #shorts #viral #facts",
            "tags": ["shorts", "facts", "viral"],
            "categoryId": "27"
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": False
        }
    },
    media_body=MediaFileUpload(video_file)
)
response = request.execute()
print("SUCCESS: Video Uploaded! ID:", response.get('id'))
