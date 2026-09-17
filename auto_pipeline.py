import os
import json
import random
import requests
import math
import numpy as np
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip, concatenate_videoclips
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

WIDTH, HEIGHT = 1080, 1920
VIDEO_PATH = "final_short.mp4"
TOKEN_PATH = "token.json"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

FACTS = [
    {
        "topic": "space",
        "script": [
            {"text": "Did you know that space is completely silent?", "query": "space galaxy dark"},
            {"text": "There is a giant cloud of alcohol floating in deep space!", "query": "nebula cosmos"},
            {"text": "A full NASA space suit costs around 12 million dollars!", "query": "astronaut space suit"}
        ]
    },
    {
        "topic": "ocean",
        "script": [
            {"text": "The ocean holds 99 percent of the living space on Earth.", "query": "deep ocean water"},
            {"text": "We know more about Mars than our ocean floor!", "query": "underwater seabed"},
            {"text": "Deep down in the ocean, there are underwater rivers and waterfalls!", "query": "underwater current ocean"}
        ]
    },
    {
        "topic": "brain",
        "script": [
            {"text": "Your brain generates enough electricity to power a small light bulb!", "query": "human brain glowing neural"},
            {"text": "It processes information at a speed of 268 miles per hour.", "query": "digital network speed light"},
            {"text": "It consumes 20 percent of your total energy!", "query": "human body energy glow"}
        ]
    }
]

def create_text_image(text):
    img = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(font_path, 42) if os.path.exists(font_path) else ImageFont.load_default()

    import textwrap
    lines = textwrap.wrap(text, width=28)
    line_height = 55
    total_text_height = len(lines) * line_height
    start_y = (HEIGHT - total_text_height) // 2

    draw.rectangle([60, start_y - 30, WIDTH - 60, start_y + total_text_height + 30], fill=(0, 0, 0, 180))

    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        x = (WIDTH - w) // 2
        y = start_y + i * line_height
        draw.text((x, y), line, font=font, fill="white")

    img.save("text_overlay.png")

def download_pexels_video(query, idx):
    PEXELS_KEY = os.getenv("PEXELS_API_KEY")
    if not PEXELS_KEY:
        return None
    try:
        headers = {"Authorization": PEXELS_KEY}
        url = f"https://api.pexels.com/videos/search?query={query}&orientation=portrait&per_page=3"
        res = requests.get(url, headers=headers).json()
        videos = res.get("videos", [])
        if videos:
            v_url = videos[0]["video_files"][0]["link"]
            fname = f"clip_{idx}.mp4"
            with open(fname, "wb") as f:
                f.write(requests.get(v_url).content)
            return fname
    except Exception as e:
        print(f"Failed download for {query}: {e}")
    return None

def generate_video():
    selected = random.choice(FACTS)
    topic = selected["topic"]
    script_segments = selected["script"]
    
    full_text = " ".join([seg["text"] for seg in script_segments])
    
    # Generate Voiceover
    tts = gTTS(text=full_text, lang='en', slow=False)
    tts.save("voiceover.mp3")
    audio = AudioFileClip("voiceover.mp3")
    total_duration = audio.duration

    segment_duration = total_duration / len(script_segments)
    clips = []

    for idx, seg in enumerate(script_segments):
        fname = download_pexels_video(seg["query"], idx)
        if fname and os.path.exists(fname):
            vc = VideoFileClip(fname).without_audio().resize((WIDTH, HEIGHT))
            clip = vc.subclip(0, min(vc.duration, segment_duration + 0.5))
            if idx > 0:
                clip = clip.crossfadein(0.5)
            clips.append(clip)

    if not clips:
        black_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        bg_clip = ImageClip(black_frame).set_duration(total_duration)
    else:
        bg_clip = concatenate_videoclips(clips, padding=-0.5, method="compose").subclip(0, total_duration)

    # Text Overlay
    create_text_image(full_text)
    txt_clip = ImageClip("text_overlay.png").set_duration(total_duration)

    # Final Composite
    final_clip = CompositeVideoClip([bg_clip, txt_clip]).set_audio(audio)
    final_clip.write_videofile(VIDEO_PATH, fps=24, codec="libx264", audio_codec="aac")
    return full_text, topic

def write_token_file():
    token_json = os.environ.get("TOKEN_JSON")
    if not token_json:
        raise RuntimeError("TOKEN_JSON variable missing")
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
            "tags": ["shorts", topic, "facts"],
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

    print(f"[upload] Done: https://youtube.com/shorts/{response['id']}")

def main():
    script_text, topic = generate_video()
    write_token_file()
    youtube = get_youtube_service()
    upload_video(youtube, script_text, topic)

if __name__ == "__main__":
    main()
