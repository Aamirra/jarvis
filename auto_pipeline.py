"""
auto_pipeline.py

End-to-end pipeline for the "jarvis" YouTube Shorts uploader.

1. Picks a random motivational quote.
2. Renders a vertical (1080x1920) quote card with Pillow.
3. Generates narration audio for the quote with gTTS.
4. Combines the image + audio into automated_daily_short.mp4 with MoviePy.
5. Authenticates with YouTube using the TOKEN_JSON GitHub secret.
6. Uploads the video as a YouTube Short.

Run with:  python auto_pipeline.py
Requires the TOKEN_JSON environment variable to be set (see daily_upload.yml).
"""

import os
import random
import textwrap

from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
from moviepy.editor import ImageClip, AudioFileClip

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
WIDTH, HEIGHT = 1080, 1920
FRAME_PATH = "frame.png"
AUDIO_PATH = "narration.mp3"
VIDEO_PATH = "automated_daily_short.mp4"
TOKEN_PATH = "token.json"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

QUOTES = [
    "The best way to predict the future is to create it.",
    "Success is not final, failure is not fatal: it is the courage to continue that counts.",
    "Your limitation, it's only your imagination.",
    "Push yourself, because no one else is going to do it for you.",
    "Great things never come from comfort zones.",
    "Dream it. Wish it. Do it.",
    "Success doesn't just find you. You have to go out and get it.",
    "The harder you work for something, the greater you'll feel when you achieve it.",
    "Don't stop when you're tired. Stop when you're done.",
    "Wake up with determination. Go to bed with satisfaction.",
    "Little things make big days.",
    "It's going to be hard, but hard does not mean impossible.",
    "Don't wait for opportunity. Create it.",
    "Sometimes we're tested not to show our weaknesses, but to discover our strengths.",
    "The key to success is to focus on goals, not obstacles.",
]


# ----------------------------------------------------------------------
# Step 1: Video generation
# ----------------------------------------------------------------------
def pick_quote() -> str:
    return random.choice(QUOTES)


def load_font(size: int):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_gradient_background() -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), "black")
    draw = ImageDraw.Draw(img)
    top = (18, 18, 40)
    bottom = (90, 30, 130)
    for y in range(HEIGHT):
        ratio = y / HEIGHT
        r = int(top[0] + (bottom[0] - top[0]) * ratio)
        g = int(top[1] + (bottom[1] - top[1]) * ratio)
        b = int(top[2] + (bottom[2] - top[2]) * ratio)
        draw.line([(0, y), (WIDTH, y)], fill=(r, g, b))
    return img


def render_quote_frame(quote: str) -> None:
    img = make_gradient_background()
    draw = ImageDraw.Draw(img)
    font = load_font(72)

    wrapped = textwrap.fill(quote, width=20)
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=22, align="center")
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (WIDTH - text_w) / 2
    y = (HEIGHT - text_h) / 2

    draw.multiline_text(
        (x, y), wrapped, font=font, fill="white", align="center", spacing=22
    )
    img.save(FRAME_PATH)


def generate_narration(quote: str) -> None:
    tts = gTTS(text=quote, lang="en")
    tts.save(AUDIO_PATH)


def build_video() -> None:
    audio_clip = AudioFileClip(AUDIO_PATH)
    duration = audio_clip.duration + 1.0  # small padding at the end

    clip = ImageClip(FRAME_PATH).set_duration(duration).set_audio(audio_clip)
    clip.write_videofile(
        VIDEO_PATH,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        verbose=False,
        logger=None,
    )


def generate_video() -> str:
    quote = pick_quote()
    print(f"[generate] Selected quote: {quote}")
    render_quote_frame(quote)
    generate_narration(quote)
    build_video()
    print(f"[generate] Video written to {VIDEO_PATH}")
    return quote


# ----------------------------------------------------------------------
# Step 2: YouTube upload
# ----------------------------------------------------------------------
def write_token_file() -> None:
    token_json = os.environ.get("TOKEN_JSON")
    if not token_json:
        raise RuntimeError(
            "TOKEN_JSON environment variable is not set. "
            "Add it as a GitHub Actions secret and pass it into this step."
        )
    with open(TOKEN_PATH, "w") as f:
        f.write(token_json)


def get_youtube_service():
    creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return build("youtube", "v3", credentials=creds)


def upload_video(youtube, quote: str) -> str:
    body = {
        "snippet": {
            "title": "Daily Motivation #Shorts",
            "description": f"{quote}\n\n#Shorts #Motivation #DailyQuote",
            "tags": ["shorts", "motivation", "quotes", "daily"],
            "categoryId": "22",
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
    print(f"[upload] Done. https://youtube.com/shorts/{video_id}")
    return video_id


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    quote = generate_video()
    write_token_file()
    youtube = get_youtube_service()
    upload_video(youtube, quote)


if __name__ == "__main__":
    main()
