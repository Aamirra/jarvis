import os
import json
import random
import requests
import math
import numpy as np
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip, concatenate_videoclips
import moviepy.video.fx.all as vfx
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
    {
        "topic": "space", 
        "text": "Did you know that space is completely silent? There is no atmosphere in space, which means sound has no way to travel to be heard. Plus, floating in deep space, there is a giant cloud of alcohol containing trillions of liters! Also, a full NASA space suit costs around 12 million dollars!"
    },
    {
        "topic": "ocean", 
        "text": "The ocean holds 99 percent of the living space on Earth, yet we have explored less than 5 percent of it. We actually know more about the surface of Mars and the Moon than our own ocean floor! Deep down in the ocean, there are underwater rivers, waterfalls, and lakes!"
    },
    {
        "topic": "brain", 
        "text": "Your brain generates about 20 watts of electricity, which is enough power to light up a dim LED bulb! It processes information at a speed of 268 miles per hour. Even though it makes up only 2 percent of your body mass, it consumes 20 percent of your total energy!"
    }
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
    
    line_height = 55
    total_text_height = len(lines) * line_height
    start_y = (HEIGHT - total_text_height) // 2

    padding = 30
    box_top = start_y - padding
    box_bottom = start_y + total_text_height + padding
    draw.rectangle([60, box_top, WIDTH - 60, box_bottom], fill=(0, 0, 0, 180))

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
    
    # 1. Voiceover & Length
    tts = gTTS(text=script_text, lang='en', slow=False)
    tts.save("voiceover.mp3")
    audio = AudioFileClip("voiceover.mp3")
    duration = audio.duration

    # Calculate clips needed for smooth flow (4 seconds each)
    CLIP_DURATION = 4.0
    num_clips_needed = math.ceil(duration / CLIP_DURATION)

    PEXELS_KEY = os.getenv("PEXELS_API_KEY")
    clips = []

    if PEXELS_KEY:
        try:
            headers = {"Authorization": PEXELS_KEY}
            url = f"https://api.pexels.com/videos/search?query={search_query}&orientation=portrait&per_page={max(15, num_clips_needed + 5)}"
            res = requests.get(url, headers=headers).json()
            videos = res.get("videos", [])
            
            downloaded_files = []
            for idx, vid in enumerate(videos[:num_clips_needed]):
                v_files = vid.get("video_files", [])
                v_url = v_files[0]["link"]
                fname = f"bg_{idx}.mp4"
                with open(fname, "wb") as f:
                    f.write(requests.get(v_url).content)
                downloaded_files.append(fname)

            # Process clips with Seamless Transitions
            padding = 0.5  # overlap for smooth crossfade
            for idx, fname in enumerate(downloaded_files):
                vc = VideoFileClip(fname).without_audio().resize((WIDTH, HEIGHT))
                c_dur = min(vc.duration, CLIP_DURATION + padding)
                clip = vc.subclip(0, c_dur)
                
                # Add crossfade to avoid jarring cuts between scenes
                if idx > 0:
                    clip = clip.crossfadein(0.5)
                clips.append(clip)

        except Exception as e:
            print(f"Pexels fetch failed: {e}")

    # Fallback to black screen if API fails
    if not clips:
        black_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        bg_clip = ImageClip(black_frame).set_duration(duration)
    else:
        # Seamlessly blend video clips together
        bg_clip = concatenate_videoclips(clips, padding=-0.5, method="compose").subclip(0, duration)

    # 3. Text Overlay
    create_text_image(script_text)
    txt_clip = ImageClip("text_overlay.png").set_duration(duration)

    # 4. Export
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
