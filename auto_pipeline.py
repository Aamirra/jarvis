import os
import json
import random
import requests
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Config
WIDTH, HEIGHT = 1080, 1920
VIDEO_PATH = "final_short.mp4"
TOKEN_PATH = "token.json"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

FACTS = [
    {"topic": "space", "text": "Did you know that space is completely silent? There is no atmosphere in space, which means sound has no way to travel. Plus, there is a giant cloud of alcohol floating in deep space!"},
    {"topic": "ocean", "text": "The ocean holds 99 percent of the living space on Earth. We have explored less than 5 percent of our oceans, meaning we know more about Mars than our own sea floor!"},
    {"topic": "brain", "text": "Your brain generates about 20 watts of electricity. That is enough power to light a dim LED bulb! It also processes information as fast as 268 miles per hour."}
]

def create_text_image(text):
    img = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if os.path.exists(font_path):
        font = ImageFont.truetype(font_path, 42)
    else:
        font = ImageFont.load_default()

    import textwrap
    lines = textwrap.wrap(text, width=28)
    
    # Calculate box height
    line_height = 55
    total_text_height = len(lines) * line_height
    start_y = (HEIGHT - total_text_height) // 2

    # Draw dark background box for text
    padding = 30
    box_top = start_y - padding
    box_bottom = start_y + total_text_height + padding
    draw.rectangle([60, box_top, WIDTH - 60, box_bottom], fill=(0, 0, 0, 180))

    # Draw text lines
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        x = (WIDTH - w) // 2
        y = start_y + i * line_height
        draw.text((x, y), line, font=font, fill="white")

    img.save("text_overlay.png")

def generate_video():
    selected = random.choice(FACTS)
    script_text = selected["text"]
    search_query = selected["topic"]
    
    # Audio
    tts = gTTS(text=script_text, lang='en', slow=False)
    tts.save("voiceover.mp3")
    audio = AudioFileClip("voiceover.mp3")
    duration = audio.duration

    # Background Video from Pexels
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
        bg_clip = ImageClip(Image.new("RGB", (WIDTH, HEIGHT), "black")).set_duration(duration)

    # Text Overlay Image
    create_text_image(script_text)
    txt_clip = ImageClip("text_overlay.png").set_duration(duration)

    # Composite & Export
    final_clip = CompositeVideoClip([bg_clip, txt_clip]).set_audio(audio)
    final_clip.write_videofile(VIDEO_PATH, fps=24, codec="libx264", audio_codec="aac")
    return script_text, search_query

def write_token_file():
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

def upload_video(youtube, description, topic):
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

def main():
    script_text, topic = generate_video()
    write_token_file()
    youtube = get_youtube_service()
    upload_video(youtube, script_text, topic)

if __name__ == "__main__":
    main()
