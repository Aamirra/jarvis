import os
import json
import time
from datetime import datetime, timezone

import requests
import numpy as np
import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"): PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS
from gtts import gTTS
from moviepy import (
    VideoFileClip, AudioFileClip, ImageClip, concatenate_videoclips,
    concatenate_audioclips, TextClip, CompositeVideoClip, AudioClip
)

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

WIDTH, HEIGHT = 1080, 1920
TARGET_DURATION = 60.0  # final video will always be exactly this long
PEXELS_KEY = os.environ.get("PEXELS_API_KEY")

# Font used for on-screen captions (installed via apt in the workflow: fonts-dejavu-core)
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

VIDEO_TITLE = "Space is Completely Silent 🤯 #shorts"
VIDEO_DESCRIPTION = (
    "Did you know space has no sound? Subscribe for a new mind-blowing "
    "space fact every single day! #space #shorts #facts #astronomy"
)
VIDEO_TAGS = ["space", "shorts", "facts", "science", "astronomy", "space facts"]


def fetch_clip(query, duration_needed, index):
    headers = {"Authorization": PEXELS_KEY}
    url = "https://api.pexels.com/videos/search?query=" + query + chr(38) + "per_page=1"
    video_file = f"bg_{index}.mp4"

    for attempt in range(3):
        try:
            time.sleep(1)
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200 and resp.json().get("videos"):
                v_files = resp.json()["videos"][0]["video_files"]
                hd_file = max(v_files, key=lambda x: x.get("width", 0))
                with open(video_file, "wb") as vf:
                    vf.write(requests.get(hd_file["link"], timeout=15).content)
                clip = VideoFileClip(video_file).without_audio()
                w, h = clip.size
                scale = HEIGHT / h
                new_w = int(w * scale)
                clip = clip.resized((new_w, HEIGHT))
                x1 = (new_w - WIDTH) // 2
                clip = clip.cropped(x1=x1, y1=0, x2=x1 + WIDTH, y2=HEIGHT)
                clips_list = []
                cur_dur = 0
                while cur_dur < duration_needed:
                    clips_list.append(clip)
                    cur_dur += clip.duration
                return concatenate_videoclips(clips_list).subclipped(0, duration_needed)
            else:
                print(f"Pexels returned status {resp.status_code} for query '{query}': {resp.text[:200]}")
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {query}: {e}")
            time.sleep(2)

    return ImageClip(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)).with_duration(duration_needed)


def add_caption(bg_clip, caption_text, duration, is_cta=False):
    """Overlay on-screen caption text on top of a background clip. Falls back
    to the plain background clip if caption rendering fails for any reason,
    so the pipeline never crashes just because of a font/text issue."""
    try:
        caption = TextClip(
            font=FONT_PATH,
            text=caption_text,
            font_size=72 if is_cta else 58,
            color="yellow" if is_cta else "white",
            stroke_color="black",
            stroke_width=3 if is_cta else 2,
            method="caption",
            size=(int(WIDTH * 0.85), None),
            text_align="center",
        ).with_duration(duration).with_position(("center", int(HEIGHT * 0.72)))

        return CompositeVideoClip([bg_clip, caption], size=(WIDTH, HEIGHT)).with_duration(duration)
    except Exception as e:
        print(f"Caption failed for text '{caption_text[:30]}...': {e}")
        return bg_clip


def get_youtube_client():
    """Build an authenticated YouTube API client from the TOKEN_JSON and
    CLIENT_SECRET_JSON secrets, refreshing the access token if needed."""
    token_raw = os.environ.get("TOKEN_JSON")
    client_raw = os.environ.get("CLIENT_SECRET_JSON")

    if not token_raw:
        raise RuntimeError("TOKEN_JSON secret is missing or empty.")
    if not client_raw:
        raise RuntimeError("CLIENT_SECRET_JSON secret is missing or empty.")

    token_data = json.loads(token_raw)
    client_data = json.loads(client_raw)
    # Google Cloud Console downloads wrap this under "installed" or "web"
    client_info = client_data.get("installed") or client_data.get("web") or client_data

    creds = Credentials(
        token=token_data.get("token"),
        refresh_token=token_data.get("refresh_token"),
        token_uri=token_data.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=token_data.get("client_id") or client_info.get("client_id"),
        client_secret=token_data.get("client_secret") or client_info.get("client_secret"),
        scopes=token_data.get("scopes", ["https://www.googleapis.com/auth/youtube.upload"]),
    )

    if not creds.valid:
        if creds.refresh_token:
            creds.refresh(Request())
        else:
            raise RuntimeError(
                "Stored credentials are invalid/expired and no refresh_token is available. "
                "You'll need to regenerate TOKEN_JSON."
            )

    return build("youtube", "v3", credentials=creds)


def upload_to_youtube(video_path):
    youtube = get_youtube_client()

    body = {
        "snippet": {
            "title": VIDEO_TITLE,
            "description": VIDEO_DESCRIPTION,
            "tags": VIDEO_TAGS,
            "categoryId": "28",  # Science & Technology
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(video_path, chunksize=-1, resumable=True, mimetype="video/mp4")
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"Upload progress: {int(status.progress() * 100)}%")

    video_id = response["id"]
    print(f"Uploaded successfully: https://youtube.com/shorts/{video_id}")
    return video_id


def update_tracker(video_id):
    tracker_file = "upload_tracker.json"
    data = []
    if os.path.exists(tracker_file):
        try:
            with open(tracker_file) as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []

    data.append({
        "video_id": video_id,
        "title": VIDEO_TITLE,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "url": f"https://youtube.com/shorts/{video_id}",
    })

    with open(tracker_file, "w") as f:
        json.dump(data, f, indent=2)


def generate_video():
    # Each scene: spoken text, Pexels search query, is this the subscribe CTA scene,
    # and an optional shorter on-screen caption (falls back to the spoken text).
    scenes = [
        {"text": "Did you know that space is completely silent?",
         "query": "deep space cosmos silence"},
        {"text": "Sound waves require a medium like air to travel, and because space is a vacuum, molecules are too far apart to carry sound.",
         "query": "space vacuum stars"},
        {"text": "If you screamed in space, no one would hear you.",
         "query": "astronaut floating space"},
        {"text": "This eerie silence stretches across the entire universe, making the cosmos both breathtaking and strangely terrifying.",
         "query": "galaxy spinning nebula"},
        {"text": "Planets, stars, and galaxies move in complete quiet, hidden behind the vastness of interstellar dark matter.",
         "query": "planet orbiting space dark"},
        {"text": "Even massive explosions like supernovas produce no sound that could ever reach your ears.",
         "query": "supernova explosion space"},
        {"text": "Scientists instead study space through light, radiation, and gravitational waves rather than sound.",
         "query": "telescope observing space"},
        {"text": "The next time you look up at the night sky, remember that all of that beauty is happening in absolute silence.",
         "query": "night sky stars milky way"},
        {"text": "If this blew your mind, hit that subscribe button and turn on notifications, because we post a brand new space fact every single day.",
         "query": "colorful nebula space bright",
         "cta": True,
         "caption": "SUBSCRIBE FOR MORE!"},
    ]

    audio_clips = []
    video_clips = []
    for i, scene in enumerate(scenes):
        fname = f"part_{i}.mp3"
        gTTS(text=scene["text"], lang="en").save(fname)
        aclip = AudioFileClip(fname)
        audio_clips.append(aclip)

        bg_clip = fetch_clip(scene["query"], aclip.duration, i)
        caption_text = scene.get("caption", scene["text"])
        scene_clip = add_caption(bg_clip, caption_text, aclip.duration, is_cta=scene.get("cta", False))
        video_clips.append(scene_clip)

    final_audio = concatenate_audioclips(audio_clips)
    final_video = concatenate_videoclips(video_clips)

    current_duration = min(final_audio.duration, final_video.duration)

    if current_duration > TARGET_DURATION:
        final_audio = final_audio.subclipped(0, TARGET_DURATION)
        final_video = final_video.subclipped(0, TARGET_DURATION)
    elif current_duration < TARGET_DURATION:
        pad = TARGET_DURATION - current_duration
        silence = AudioClip(lambda t: 0, duration=pad, fps=44100)
        final_audio = concatenate_audioclips([final_audio, silence])

        last_frame = final_video.get_frame(max(final_video.duration - 0.04, 0))
        freeze = ImageClip(last_frame).with_duration(pad)
        final_video = concatenate_videoclips([final_video, freeze])

    final_duration = min(final_audio.duration, final_video.duration)
    final_audio = final_audio.subclipped(0, final_duration)
    final_video = final_video.subclipped(0, final_duration)

    final = final_video.with_audio(final_audio)
    output_path = "final_short.mp4"
    final.write_videofile(output_path, fps=30, codec="libx264", audio_codec="aac", bitrate="5000k")

    video_id = upload_to_youtube(output_path)
    update_tracker(video_id)


if __name__ == "__main__":
    generate_video()
