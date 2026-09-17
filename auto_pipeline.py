import os
import json
import random
import requests
from gtts import gTTS
from moviepy.editor import TextClip, CompositeVideoClip, AudioFileClip, VideoFileClip
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# ----------------------------------------------------------------------
# Config & Data
# ----------------------------------------------------------------------
WIDTH, HEIGHT = 1080, 1920
VIDEO_PATH = "final_short.mp4"
TOKEN_PATH = "token.json"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

FACTS = [
    {"topic": "space", "text": "Did you know that space is completely silent? There is no atmosphere in space, which means sound has no way to travel. Plus, there is a giant cloud of alcohol floating in deep space!"},
    {"topic": "ocean", "text": "The ocean holds 99 percent of the living space on Earth. We have explored less than 5 percent of our oceans, meaning we know more about Mars than our own sea floor!"},
    {"topic": "brain", "text": "Your brain generates about 20 watts of electricity. That is enough power to light a dim LED bulb! It also processes information as fast as 268 miles per hour."}
]

# ----------------------------------------------------------------------
# Step 1: Video & Audio Generation (Pexels + MoviePy)
# ----------------------------------------------------------------------
def generate_video():
    selected = random.choice(FACTS)
    script_text = selected["text"]
    search_query = selected["topic"]
    
    # 1. Voiceover
    tts = gTTS(text=script_text, lang='en', slow=False)
    tts.save("voiceover.mp3")
    audio = AudioFileClip("voiceover.mp3")
    duration = audio.duration

    # 2. Pexels Background Video
    PEXELS_KEY = os.getenv("PEXELS_API_KEY")
    video_file = "bg_video.mp4"
    bg_clip = None

    if PEXELS_KEY:
        try:
            headers = {"Authorization": PEXELS_KEY}
            url = f"https://api.pexels.com/videos/search?query={search_query}&orientation=portrait&per_page=5"
            res = requests.get(url, headers=headers).json()
            if res.get("videos"):
                video_url = res["videos"][0]["video_files"][0]["link"]
                with open(video_file, "wb") as f:
                    f.write(requests.get(video_url).content)
                bg_clip = VideoFileClip(video_file).subclip(0, min(duration, 20)).resize((WIDTH, HEIGHT))
        except Exception as e:
            print(f"Pexels fetch failed: {e}")

    if bg_clip is None:
        bg_clip = TextClip("", size=(WIDTH, HEIGHT), bg_color="black").set_duration(duration)

    # 3. Dynamic Captions
    txt_clip = TextClip(
        script_text, 
        fontsize=45, 
        color='white', 
        bg_color='black',
        method='caption', 
        size=(880, None)
    ).set_duration(duration).set_position('center')

    # 4. Composite & Export
    final_clip = CompositeVideoClip([bg_clip, txt_clip]).set_audio(audio)
    final_clip.write_videofile(VIDEO_PATH, fps=24, codec="libx264", audio_codec="aac")
    return script_text, search_query

# ----------------------------------------------------------------------
# Step 2: YouTube Upload
# ----------------------------------------------------------------------
def write_token_file() -> None:
    token_json = os.environ.get("TOKEN_JSON")
    if not token_json:
        raise RuntimeError("TOKEN_JSON environment variable is not set.")
    with open(TOKEN_PATH, "w") as f:
        f.write(token_json)

def get_youtube_service():
    creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return build("youtube", "v3", credentials=creds)

def upload_video(youtube, description: str, topic: str):
    body = {
        "snippet": {
            "title": f"Mind Blowing {topic.capitalize()} Facts! #Shorts",
            "description": f"{description}\n\n#Shorts #{topic.capitalize()} #Facts #AI",
            "tags": ["shorts", topic, "facts", "educational"],
            "categoryId": "27",
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": False,
        },
    }
    media = MediaFileUpload(VIDEO_PATH, chunksize=-1, resumable=True, mimetype="video/mp4")
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"[upload] {int(status.progress() * 100)}% uploaded")

    video_id = response["id"]
    print(f"[upload] Done: https://youtube.com/shorts/{video_id}")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    script_text, topic = generate_video()
    write_token_file()
    youtube = get_youtube_service()
    upload_video(youtube, script_text, topic)

if __name__ == "__main__":
    main()
