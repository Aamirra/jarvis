import os
import json
import random
import time
from datetime import datetime, timezone

import requests
import numpy as np
import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"): PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS
from gtts import gTTS
from moviepy import (
    VideoFileClip, AudioFileClip, ImageClip, concatenate_videoclips,
    concatenate_audioclips, TextClip, CompositeVideoClip, AudioClip,
    CompositeAudioClip
)

from google import genai
from google.genai import types

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

WIDTH, HEIGHT = 1080, 1920
TARGET_DURATION = 60.0  # final video will always be exactly this long
PEXELS_KEY = os.environ.get("PEXELS_API_KEY")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.1-flash-lite"  # cheap + fast, plenty for short scripts

TRACKER_FILE = "upload_tracker.json"
MUSIC_DIR = "music"  # put a few royalty-free .mp3 files here; one is picked at random each run

# Font used for on-screen captions (installed via apt in the workflow: fonts-dejavu-core)
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

CTA_SCENE = {
    "text": "If this blew your mind, hit that subscribe button and turn on notifications, because we post brand new videos every single day.",
    "query": "colorful nebula space bright",
    "cta": True,
    "caption": "SUBSCRIBE FOR MORE!",
}

# Used only if AI script generation fails, so the pipeline never crashes.
FALLBACK_SCRIPTS = {
    "random": {
        "title": "Space is Completely Silent",
        "hashtags": ["space", "facts", "didyouknow", "science", "universe"],
        "scenes": [
            {"text": "Did you know that space is completely silent?", "query": "deep space cosmos silence"},
            {"text": "Sound waves require a medium like air to travel, and because space is a vacuum, molecules are too far apart to carry sound.", "query": "space vacuum stars"},
            {"text": "If you screamed in space, no one would hear you.", "query": "astronaut floating space"},
            {"text": "This eerie silence stretches across the entire universe, making the cosmos both breathtaking and strangely terrifying.", "query": "galaxy spinning nebula"},
            {"text": "Planets, stars, and galaxies move in complete quiet, hidden behind the vastness of interstellar dark matter.", "query": "planet orbiting space dark"},
        ],
    },
    "ai_tips": {
        "title": "This One Prompt Trick Changes Everything",
        "hashtags": ["ai", "aitips", "chatgpt", "prompting", "productivity"],
        "scenes": [
            {"text": "Most people use AI chatbots completely wrong, and it's costing them much better answers.", "query": "person typing laptop screen"},
            {"text": "Here's a simple trick: instead of just asking a question, show the AI an example of what you want first.", "query": "hands typing keyboard closeup"},
            {"text": "This is called few shot prompting, and it works because AI learns better from examples than from instructions alone.", "query": "digital technology abstract lights"},
            {"text": "For example, instead of asking for a good title, show it two titles you already like, then ask for a third in that same style.", "query": "notebook writing ideas desk"},
            {"text": "Try this in your very next chat with any AI tool, and watch how much better the answer gets.", "query": "smartphone chat app screen"},
        ],
    },
}


def choose_content_type():
    """Decide today's content type from the UTC hour, so the 4 daily runs
    (every 6 hours) split evenly into 2 random-topic videos and 2 AI-tips
    videos without needing any extra state file."""
    hour = datetime.now(timezone.utc).hour
    if hour in (6, 18):
        return "ai_tips"
    return "random"  # covers hour 0, 12, and any manual/off-schedule run


def get_past_titles(limit=20):
    """Read titles of previously uploaded videos so we can ask the AI to
    avoid repeating the same topic."""
    if not os.path.exists(TRACKER_FILE):
        return []
    try:
        with open(TRACKER_FILE) as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        return [entry.get("title", "") for entry in data[-limit:] if entry.get("title")]
    except Exception:
        return []


def build_prompt(content_type, avoid_text):
    if content_type == "ai_tips":
        return f"""You write short, punchy scripts for a YouTube Shorts channel
that teaches everyday people practical AI tips, tricks, and beginner concepts
for using AI chatbots and tools (like ChatGPT, Gemini, Claude, or similar) in
daily life, work, or study. Assume the viewer is a curious beginner, not a
programmer.

{avoid_text}

Return ONLY valid JSON in exactly this shape, no extra commentary:
{{
  "title": "a short catchy title for the video, under 8 words",
  "hashtags": ["5 to 8 relevant lowercase hashtags for this specific video, no # symbol"],
  "scenes": [
    {{"text": "one or two spoken sentences", "query": "2-4 word English stock-footage search term for this sentence"}}
  ]
}}

Rules:
- 6 to 8 scenes total.
- Teach ONE genuinely useful, concrete AI tip, trick, or concept per video (e.g. a prompting technique, a way to save time, a common mistake to avoid, or a simple explanation of how AI works).
- All scenes combined should read aloud in about 40-45 seconds (roughly 110-140 words total).
- Start with a hook about a mistake or surprising fact, then explain the tip clearly, then give one short concrete example.
- Each "query" must describe generic stock video footage (people using devices, offices, technology, abstract digital visuals) - never named apps' logos or real people - so it can be found on a stock footage site.
- hashtags should mix a couple of broad/high-traffic tags (like "ai", "shorts") with a few specific to this exact tip, to help discovery.
- Keep language simple, practical, and conversational, suitable for text-to-speech narration.
"""
    else:
        return f"""You write short, punchy scripts for a "did you know" style
YouTube Shorts channel about surprising true facts (space, science, history,
psychology, nature, animals, or the human body - pick ONE topic at random,
something genuinely surprising and different each time).

{avoid_text}

Return ONLY valid JSON in exactly this shape, no extra commentary:
{{
  "title": "a short catchy title for the video, under 8 words",
  "hashtags": ["5 to 8 relevant lowercase hashtags for this specific video, no # symbol"],
  "scenes": [
    {{"text": "one or two spoken sentences", "query": "2-4 word English stock-footage search term for this sentence"}}
  ]
}}

Rules:
- 6 to 8 scenes total.
- All scenes combined should read aloud in about 40-45 seconds (roughly 110-140 words total).
- Each scene's spoken text should flow into the next like a mini story with a hook, build-up, and a surprising payoff.
- Each "query" must describe generic stock video footage (nature, objects, places, animals) - never named people or brands - so it can be found on a stock footage site.
- hashtags should mix a couple of broad/high-traffic tags (like "shorts", "facts") with a few specific to this exact topic, to help discovery.
- Keep language simple and conversational, suitable for text-to-speech narration.
"""


def generate_script_with_ai(content_type, max_attempts=3):
    if not GEMINI_KEY:
        print("GEMINI_API_KEY not set, using fallback script.")
        return FALLBACK_SCRIPTS[content_type]

    past_titles = get_past_titles()
    avoid_text = ""
    if past_titles:
        avoid_text = (
            "Do NOT repeat these topics already covered, pick something different: "
            + "; ".join(past_titles)
        )

    prompt = build_prompt(content_type, avoid_text)

    try:
        client = genai.Client(api_key=GEMINI_KEY)
    except Exception as e:
        print(f"Could not create Gemini client, using fallback script: {type(e).__name__}: {e}")
        return FALLBACK_SCRIPTS[content_type]

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )

            raw_text = getattr(response, "text", None)
            if not raw_text:
                # Often means the response was empty or blocked by safety filters
                raise ValueError(
                    f"Empty response from Gemini (prompt_feedback={getattr(response, 'prompt_feedback', None)})"
                )

            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].strip()

            data = json.loads(cleaned)

            if not data.get("title") or not data.get("scenes"):
                raise ValueError("AI response JSON is missing 'title' or 'scenes'.")
            for scene in data["scenes"]:
                if not scene.get("text") or not scene.get("query"):
                    raise ValueError("A scene in the AI response is missing 'text' or 'query'.")

            print(f"Gemini script generated successfully on attempt {attempt}/{max_attempts}.")
            return data

        except Exception as e:
            last_error = e
            print(f"[Attempt {attempt}/{max_attempts}] Gemini script generation failed: {type(e).__name__}: {e}")
            if attempt < max_attempts:
                time.sleep(3 * attempt)  # brief backoff before retrying

    print(f"All {max_attempts} Gemini attempts failed, using fallback script. Last error: {last_error}")
    return FALLBACK_SCRIPTS[content_type]


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


def add_title_flash(scene_clip, title_text, flash_duration=1.5):
    """Overlay a big bold title card for the first ~1.5 seconds of the very
    first scene, as a stronger scroll-stopping hook. Falls back to the plain
    scene clip if anything goes wrong."""
    try:
        flash_len = min(flash_duration, scene_clip.duration)
        title_clip = TextClip(
            font=FONT_PATH,
            text=title_text.upper(),
            font_size=80,
            color="white",
            stroke_color="black",
            stroke_width=4,
            method="caption",
            size=(int(WIDTH * 0.9), None),
            text_align="center",
        ).with_duration(flash_len).with_position(("center", "center"))

        return CompositeVideoClip([scene_clip, title_clip], size=(WIDTH, HEIGHT)).with_duration(scene_clip.duration)
    except Exception as e:
        print(f"Title flash failed: {e}")
        return scene_clip


def add_background_music(narration_audio, duration):
    """Mix a quiet, looped royalty-free track under the narration. Put a few
    .mp3 files in a 'music' folder in the repo - one is picked at random each
    run. If the folder is missing or empty, narration plays with no music."""
    if not os.path.isdir(MUSIC_DIR):
        print("No 'music' folder found, skipping background music.")
        return narration_audio

    tracks = [f for f in os.listdir(MUSIC_DIR) if f.lower().endswith((".mp3", ".wav", ".m4a"))]
    if not tracks:
        print("'music' folder is empty, skipping background music.")
        return narration_audio

    try:
        track_path = os.path.join(MUSIC_DIR, random.choice(tracks))
        music = AudioFileClip(track_path)

        loop_clips = []
        cur_dur = 0
        while cur_dur < duration:
            loop_clips.append(music)
            cur_dur += music.duration
        music_full = concatenate_audioclips(loop_clips).subclipped(0, duration)
        music_quiet = music_full.with_volume_scaled(0.15)  # keep narration clearly audible

        return CompositeAudioClip([narration_audio, music_quiet])
    except Exception as e:
        print(f"Background music failed, continuing without it: {e}")
        return narration_audio


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


def upload_to_youtube(video_path, title, description, tags):
    youtube = get_youtube_client()

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": "27",  # Education
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


def update_tracker(video_id, title, content_type):
    data = []
    if os.path.exists(TRACKER_FILE):
        try:
            with open(TRACKER_FILE) as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []

    data.append({
        "video_id": video_id,
        "title": title,
        "content_type": content_type,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "url": f"https://youtube.com/shorts/{video_id}",
    })

    with open(TRACKER_FILE, "w") as f:
        json.dump(data, f, indent=2)


def generate_video():
    content_type = choose_content_type()
    script = generate_script_with_ai(content_type)
    title = script["title"]
    scenes = list(script["scenes"]) + [CTA_SCENE]

    print(f"Content type: {content_type} | Today's topic: {title}")

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
        if i == 0:
            scene_clip = add_title_flash(scene_clip, title)
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

    final_audio = add_background_music(final_audio, final_duration)

    final = final_video.with_audio(final_audio)
    output_path = "final_short.mp4"
    final.write_videofile(output_path, fps=30, codec="libx264", audio_codec="aac", bitrate="5000k")

    if content_type == "ai_tips":
        base_tags = ["ai", "aitips", "chatgpt", "productivity", "shorts"]
    else:
        base_tags = ["shorts", "facts", "didyouknow"]

    ai_hashtags = [h.strip().lstrip("#").lower() for h in script.get("hashtags", []) if h.strip()]
    all_tags = list(dict.fromkeys(ai_hashtags + base_tags))  # AI's topic-specific tags first, deduped

    hashtag_line = " ".join(f"#{t}" for t in all_tags[:8])
    description = f"{title}\n\nSubscribe for a new video every single day!\n\n{hashtag_line}"

    video_id = upload_to_youtube(output_path, title, description, all_tags)
    update_tracker(video_id, title, content_type)


if __name__ == "__main__":
    generate_video()
