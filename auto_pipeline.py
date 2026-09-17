import os
import glob
import requests
import whisper
import pyttsx3
import json
from datetime import date
from moviepy import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip, ColorClip
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# --- TRACKER ---
TRACK_FILE = "upload_tracker.json"
today = str(date.today())
tracker = {"date": today, "count": 0}
if os.path.exists(TRACK_FILE):
    try:
        with open(TRACK_FILE, "r") as f:
            data = json.load(f)
            if data.get("date") == today:
                tracker = data
    except Exception:
        pass

if tracker["count"] >= 3:
    print(f"[LIMIT REACHED] Aaj ke 3 Shorts poore ho chuke hain ({today}).")
    exit()

# --- SCRIPT ---
print("[1/5] Generating script...")
script_text = ""
try:
    response = requests.post('http://localhost:11434/api/generate', json={
        'model': 'llama3.2',
        'prompt': 'Write a short 30-second viral Youtube Short script about 3 shocking unknown facts. Keep it strictly under 50 words, plain text only, no headings, no stage directions.',
        'stream': False
    }, timeout=5)
    if response.status_code == 200:
        script_text = response.json().get('response', '').strip()
except Exception:
    pass

if not script_text:
    script_text = "3 mind blowing space facts. Space is completely silent. There is a giant cloud of alcohol in space. One day on Venus is longer than a year."

# --- AUDIO & CAPTIONS ---
engine = pyttsx3.init()
engine.save_to_file(script_text, 'voiceover.mp3')
engine.runAndWait()
audio = AudioFileClip("voiceover.mp3")

model = whisper.load_model("base")
result = model.transcribe("voiceover.mp3")

# --- VIDEO ---
video = ColorClip(size=(1080, 1920), color=(15, 15, 30), duration=audio.duration).with_audio(audio)
subtitles = []
for segment in result['segments']:
    txt_clip = TextClip(text=segment['text'], font_size=48, color='yellow', font='C:\\\\Windows\\\\Fonts\\\\arial.ttf', method='caption', size=(800, None))
    txt_clip = txt_clip.with_position(('center', 'center')).with_start(segment['start']).with_end(segment['end'])
    subtitles.append(txt_clip)

CompositeVideoClip([video] + subtitles).write_videofile("automated_daily_short.mp4", fps=24)

# --- AUTO FIND SECRET FILE ---
print("[5/5] Searching for OAuth Secret File...")
possible_paths = glob.glob("C:/Users/HP/Desktop/client_secret*.json") + glob.glob("C:/Users/HP/client_secret*.json")
if not possible_paths:
    raise FileNotFoundError("client_secret file nahi mili!")

secret_file = possible_paths[0]
print(f"Found Secret File: {secret_file}")

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
flow = InstalledAppFlow.from_client_secrets_file(secret_file, SCOPES)
credentials = flow.run_local_server(port=0)
youtube = build("youtube", "v3", credentials=credentials)

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

tracker["count"] += 1
with open(TRACK_FILE, "w") as f:
    json.dump(tracker, f)
