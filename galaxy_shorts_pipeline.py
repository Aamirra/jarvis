import os
import io
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
    ImageClip, VideoClip, AudioFileClip, concatenate_videoclips,
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
MIN_DURATION = 30.0
MAX_DURATION = 40.0  # each video's final length is picked randomly between these
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")

TEXT_MODEL = "gemini-2.5-flash"  # trend research + story writing (free tier is plenty for this)

# Cartoon images come from Pollinations' free, keyless image endpoint instead
# of a paid/rate-limited Gemini image model. No API key, no daily quota - the
# only rule is roughly one request every 15 seconds on the anonymous tier, so
# generate_cartoon_image() paces itself after every request, success or not.
POLLINATIONS_URL = "https://image.pollinations.ai/prompt/{prompt}"
POLLINATIONS_PACING_SECONDS = 15

# How much bigger than the final frame each scene image is generated, so
# build_ken_burns_clip() has real room to pan/zoom across it without ever
# revealing an edge of the source image.
PAN_MARGIN = 0.25
KEN_BURNS_ZOOM_RATIO = 0.22  # how much extra zoom is added over a scene's duration

TRACKER_FILE = "cartoon_upload_tracker.json"
MUSIC_DIR = "music"  # put a few royalty-free .mp3 files here; one is picked at random each run

# Font used for on-screen captions (installed via apt in the workflow: fonts-dejavu-core).
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# 6 runs per day, one every 4 hours (UTC), alternating English / Roman-Urdu
# captions. Update your workflow's cron to '0 */4 * * *' so it actually
# triggers at these hours.
RUN_SCHEDULE = {
    0: "ur",
    4: "ur",
    8: "ur",
    12: "ur",
    16: "ur",
    20: "ur",
}


def get_run_config():
    """Decide today's caption language from the UTC hour. Snaps to the
    nearest scheduled hour so a slightly delayed workflow run (common with
    GitHub Actions cron) still picks a sensible slot instead of crashing."""
    hour = datetime.now(timezone.utc).hour
    closest_hour = min(RUN_SCHEDULE.keys(), key=lambda h: min(abs(hour - h), 24 - abs(hour - h)))
    return RUN_SCHEDULE[closest_hour]


# Urdu runs: "text" stays in proper Urdu script because gTTS needs real Urdu
# script to pronounce it correctly - that field is only ever used for the
# voice-over, never shown on screen. "caption" is Roman Urdu, so anyone can
# read the on-screen text, not just people who read the Urdu script.
CTA_SCENE_EN = {
    "text": "If you liked this story, hit subscribe and turn on notifications, because we post a brand new cartoon every single day.",
    "cta": True,
    "caption": "SUBSCRIBE FOR MORE!",
}

CTA_SCENE_UR = {
    "text": "اگر یہ کہانی پسند آئی تو سبسکرائب کریں اور نوٹیفکیشن آن کریں، کیونکہ ہم روزانہ نئی کارٹون ویڈیو پوسٹ کرتے ہیں۔",
    "cta": True,
    "caption": "SUBSCRIBE KAREIN!",
}

# The 3 visual beats used for the CTA scene's little wave animation
# (start / middle / end), same idea as the AI-written scenes below.
CTA_IMAGE_BEATS = [
    "the character starting to raise one hand up, big happy smile, beginning to wave at the viewer",
    "the character mid-wave with an even bigger smile, a few sparkle icons starting to appear around them",
    "the character giving one big enthusiastic wave at the viewer, surrounded by a sparkle of subscribe/bell/like icons",
]

# Used only if AI script generation fails completely, so the pipeline never
# crashes. One safe, original (non-copyrighted) space/galaxy cartoon story.
# Each scene has 3 "image_prompts" beats (start/middle/end) so it animates
# the same way an AI-written scene would.
FALLBACK_SCRIPT = {
    "en": {
        "title": "Orbit The Planet Makes New Friends",
        "hashtags": ["cartoon", "shorts", "animation", "space", "galaxy", "planets", "kidsstory"],
        "character_sheet": "A small round planet character with big friendly eyes, a soft pastel blue-green surface with gentle swirl patterns, a tiny ring like a scarf, cute expressive smiling face, flat 2D cartoon style",
        "scenes": [
            {"text": "Orbit the little planet spun alone in his corner of the galaxy, wishing he had someone to shine with.",
             "image_prompts": [
                 "the small round planet character floating alone at the edge of a quiet galaxy, distant twinkling stars, soft light pastel background",
                 "the planet character slowly drifting further, looking around hopefully but seeing no one nearby, soft pastel space background",
                 "the planet character sighing softly, a single little star twinkling faintly in the distance, light pastel colors",
             ]},
            {"text": "One day a friendly comet zoomed by and invited Orbit to visit the sparkling star cluster nearby.",
             "image_prompts": [
                 "a cheerful comet with a colorful pastel tail zooming into view near the planet character, light bright space background",
                 "the comet circling playfully around the planet character with an inviting gesture, bright pastel colors",
                 "the planet character smiling and starting to follow the comet toward a distant sparkling star cluster, light pastel background",
             ]},
            {"text": "At the star cluster, dozens of twinkling stars welcomed Orbit and taught him how to glow even brighter.",
             "image_prompts": [
                 "the planet character arriving at a cluster of twinkling stars, the stars turning to look at them warmly, bright pastel colors",
                 "the twinkling stars gathering closer around the planet character with teaching gestures, everyone glowing softly, light pastel background",
                 "the planet character glowing brighter than before, surrounded by smiling stars, warm bright pastel colors",
             ]},
            {"text": "Now Orbit lights up the whole galaxy with his new friends, and space never feels lonely again.",
             "image_prompts": [
                 "the planet character and star friends starting to line up together across the sky, bright pastel colors",
                 "the planet character and stars glowing together, forming a colorful trail across the galaxy, sunny pastel colors",
                 "the whole galaxy lit up brightly with the planet character and star friends celebrating together, joyful sunny pastel colors",
             ]},
        ],
    },
    "ur": {
        "title": "Orbit Planet Ke Naye Dost",
        "hashtags": ["cartoon", "shorts", "animation", "space", "galaxy", "planets", "kidsstory"],
        "character_sheet": "A small round planet character with big friendly eyes, a soft pastel blue-green surface with gentle swirl patterns, a tiny ring like a scarf, cute expressive smiling face, flat 2D cartoon style",
        "scenes": [
            {"text": "اوربٹ، ایک چھوٹا سا سیارہ، کہکشاں کے ایک کونے میں اکیلا گھومتا تھا اور چاہتا تھا کہ اس کے ساتھ کوئی چمکے۔",
             "caption_roman": "Orbit, aik chota sa sayyara, kehkashan ke aik kone mein akela ghoomta tha aur chahta tha ke uske sath koi chamke.",
             "image_prompts": [
                 "the small round planet character floating alone at the edge of a quiet galaxy, distant twinkling stars, soft light pastel background",
                 "the planet character slowly drifting further, looking around hopefully but seeing no one nearby, soft pastel space background",
                 "the planet character sighing softly, a single little star twinkling faintly in the distance, light pastel colors",
             ]},
            {"text": "ایک دن ایک دوستانہ دم دار ستارہ اس کے پاس سے گزرا اور اسے قریبی چمکتے ستاروں کے جھرمٹ میں آنے کی دعوت دی۔",
             "caption_roman": "Aik din aik dostana dumdar sitara uske pass se guzra aur usay qareebi chamakte sitaron ke jhurmat mein aane ki dawat di.",
             "image_prompts": [
                 "a cheerful comet with a colorful pastel tail zooming into view near the planet character, light bright space background",
                 "the comet circling playfully around the planet character with an inviting gesture, bright pastel colors",
                 "the planet character smiling and starting to follow the comet toward a distant sparkling star cluster, light pastel background",
             ]},
            {"text": "ستاروں کے جھرمٹ میں درجنوں چمکتے ستاروں نے اوربٹ کا استقبال کیا اور اسے مزید چمکنا سکھایا۔",
             "caption_roman": "Sitaron ke jhurmat mein darjanon chamakte sitaron ne Orbit ka istaqbal kiya aur usay mazeed chamakna sikhaya.",
             "image_prompts": [
                 "the planet character arriving at a cluster of twinkling stars, the stars turning to look at them warmly, bright pastel colors",
                 "the twinkling stars gathering closer around the planet character with teaching gestures, everyone glowing softly, light pastel background",
                 "the planet character glowing brighter than before, surrounded by smiling stars, warm bright pastel colors",
             ]},
            {"text": "اب اوربٹ اپنے نئے دوستوں کے ساتھ پوری کہکشاں کو روشن کرتا ہے، اور خلا کبھی تنہا محسوس نہیں ہوتا۔",
             "caption_roman": "Ab Orbit apne naye doston ke sath puri kehkashan ko roshan karta hai, aur khala kabhi tanha mehsoos nahi hota.",
             "image_prompts": [
                 "the planet character and star friends starting to line up together across the sky, bright pastel colors",
                 "the planet character and stars glowing together, forming a colorful trail across the galaxy, sunny pastel colors",
                 "the whole galaxy lit up brightly with the planet character and star friends celebrating together, joyful sunny pastel colors",
             ]},
        ],
    },
}

STYLE_PREFIX = (
    "Flat 2D vector cartoon illustration set in outer space, BRIGHT and "
    "LIGHT color palette (soft pastel pink, light blue, lavender, warm "
    "cream, sunny yellow), light sky background - NOT a dark or black "
    "background, glowing stars, cute smiling planets, clean bold outlines, "
    "simple shapes, portrait composition, no text or watermarks anywhere "
    "in the image. "
)


def get_past_titles(language, limit=20):
    """Read titles of previously uploaded videos in the same language, so we
    can ask the AI to avoid repeating the same topic/story."""
    if not os.path.exists(TRACKER_FILE):
        return []
    try:
        with open(TRACKER_FILE) as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        matching = [e for e in data if e.get("language", "en") == language]
        return [entry.get("title", "") for entry in matching[-limit:] if entry.get("title")]
    except Exception:
        return []


def get_trending_inspiration():
    """Ask Gemini, with live Google Search grounding, what's actually
    trending in space/astronomy short-form video right now. Returns a short
    plain-text list, or None if search/AI isn't available - the script
    still works fine without it, it just falls back to an evergreen story."""
    if not GEMINI_KEY:
        return None
    try:
        client = genai.Client(api_key=GEMINI_KEY)
        response = client.models.generate_content(
            model=TEXT_MODEL,
            contents=(
                "Search for what is trending in short-form video (YouTube "
                "Shorts / TikTok / Reels) right now, today, related to "
                "space, astronomy, galaxies, or planets. Reply with a "
                "plain numbered list of 5 short, family-friendly themes, "
                "topics, or space facts that are currently popular and "
                "could realistically inspire a cute original animated "
                "cartoon short about galaxies or planets (a fun space "
                "fact, a relatable situation, a satisfying twist, or a "
                "simple lesson). No branded characters, no real people. "
                "No explanations, just the list."
            ),
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
        text = getattr(response, "text", None)
        return text.strip() if text else None
    except Exception as e:
        print(f"Trend lookup failed, continuing without it: {type(e).__name__}: {e}")
        return None


def build_prompt(avoid_text, trend_text, language):
    trend_block = ""
    if trend_text:
        trend_block = (
            "Here is what's genuinely trending in short-form video right "
            "now - loosely draw inspiration from ONE of these if it fits a "
            "cute galaxy/planet cartoon story naturally, but don't force it "
            "if none fit:\n"
            f"{trend_text}\n\n"
        )

    story_rules = """- Every story must be set in space and centered on galaxies, planets, stars, or the solar system - no non-space settings.
- 4 to 6 short scenes forming ONE complete mini story with a clear beginning, middle, and a satisfying end (a fun twist, a heartwarming moment, or a simple space fact woven into the story - whatever fits best).
- Invent ONE simple, original, appealing cartoon character themed around space - a cute planet, star, comet, moon, or friendly alien/astronaut - NEVER a real branded or copyrighted character.
- The very first sentence must be a bold, scroll-stopping hook - a surprising situation or question about space - written to grab attention in the first 2 seconds.
- All scenes combined should read aloud in about 22-30 seconds (roughly 60-85 words total).
- "character_sheet": a short, vivid visual description of the character's appearance (type, colors, accessories, expression) written for an AI image generator, detailed enough to stay visually consistent scene to scene. Always written in English.
- Each scene needs "image_prompts": an array of EXACTLY 3 short prompts describing that scene's action as 3 distinct beats - a clear start, middle, and end of whatever happens in that scene (e.g. the character reaching out, then touching something, then reacting) - so they play like a tiny 3-frame flipbook, not 3 versions of the same static pose. Keep the setting consistent across the 3 beats, only the action/pose changes. Each beat must describe a BRIGHT, LIGHT-colored space scene (soft pastel nebula/sky, NOT a black or dark background). Always written in English.
- hashtags should mix broad/high-traffic tags (like "cartoon", "shorts", "animation", "space") with a few specific to this exact story (e.g. a specific planet or galaxy name if relevant)."""

    if language == "ur":
        return f"""You write short, punchy scripts for an original animated
cartoon-shorts YouTube channel about galaxies and planets. This video is for
Urdu-speaking viewers, but captions must be readable by anyone, including
people who don't read Urdu script.

{trend_block}{avoid_text}

- "title": a short, catchy title written in ROMAN URDU (Urdu typed with English/Latin letters) - NOT Urdu script, NOT Hindi Devanagari.
- "text" (per scene): the SAME sentence written in proper URDU SCRIPT (Nastaliq/Arabic script), used only to generate the voice-over so it must be correct natural Urdu - never shown on screen.
- "caption_roman" (per scene): the SAME sentence transliterated into ROMAN URDU. This is what appears as the on-screen caption.

Return ONLY valid JSON in exactly this shape, no extra commentary:
{{
  "title": "short catchy title in Roman Urdu, under 8 words",
  "hashtags": ["5 to 8 relevant lowercase English hashtags, no # symbol"],
  "character_sheet": "visual description of the character, in English",
  "scenes": [
    {{"text": "one or two spoken sentences in proper Urdu script", "caption_roman": "the same sentences in Roman Urdu", "image_prompts": ["start beat, in English", "middle beat, in English", "end beat, in English"]}}
  ]
}}

Rules:
{story_rules}
- Keep language simple and conversational, suitable for text-to-speech narration.
"""
    else:
        return f"""You write short, punchy scripts for an original animated
cartoon-shorts YouTube channel about galaxies and planets, in English.

{trend_block}{avoid_text}

Return ONLY valid JSON in exactly this shape, no extra commentary:
{{
  "title": "a short catchy title for the video, under 8 words",
  "hashtags": ["5 to 8 relevant lowercase hashtags for this specific story, no # symbol"],
  "character_sheet": "visual description of the character",
  "scenes": [
    {{"text": "one or two spoken sentences", "image_prompts": ["start beat", "middle beat", "end beat"]}}
  ]
}}

Rules:
{story_rules}
- Keep language simple and conversational, suitable for text-to-speech narration.
"""


def generate_script_with_ai(language, trend_text, max_attempts=3):
    if not GEMINI_KEY:
        print("GEMINI_API_KEY not set, using fallback script.")
        return FALLBACK_SCRIPT[language]

    past_titles = get_past_titles(language)
    avoid_text = ""
    if past_titles:
        avoid_text = (
            "Do NOT repeat these stories/topics already covered, invent a "
            "different character and a different story: " + "; ".join(past_titles)
        )

    prompt = build_prompt(avoid_text, trend_text, language)

    try:
        client = genai.Client(api_key=GEMINI_KEY)
    except Exception as e:
        print(f"Could not create Gemini client, using fallback script: {type(e).__name__}: {e}")
        return FALLBACK_SCRIPT[language]

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=TEXT_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )

            raw_text = getattr(response, "text", None)
            if not raw_text:
                raise ValueError(
                    f"Empty response from Gemini (prompt_feedback={getattr(response, 'prompt_feedback', None)})"
                )

            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].strip()

            data = json.loads(cleaned)

            if not data.get("title") or not data.get("scenes") or not data.get("character_sheet"):
                raise ValueError("AI response JSON is missing 'title', 'character_sheet' or 'scenes'.")
            for scene in data["scenes"]:
                if not scene.get("text") or not scene.get("image_prompts"):
                    raise ValueError("A scene in the AI response is missing 'text' or 'image_prompts'.")
                if not isinstance(scene["image_prompts"], list) or len(scene["image_prompts"]) < 2:
                    raise ValueError("A scene's 'image_prompts' must be a list of at least 2 beats.")
                if language == "ur" and not scene.get("caption_roman"):
                    raise ValueError("A scene in the AI response is missing 'caption_roman'.")

            print(f"Gemini script generated successfully on attempt {attempt}/{max_attempts}.")
            return data

        except Exception as e:
            last_error = e
            print(f"[Attempt {attempt}/{max_attempts}] Gemini script generation failed: {type(e).__name__}: {e}")
            if attempt < max_attempts:
                time.sleep(3 * attempt)

    print(f"All {max_attempts} Gemini attempts failed, using fallback script. Last error: {last_error}")
    return FALLBACK_SCRIPT[language]


def fit_to_size(img, target_w, target_h):
    """Resize + center-crop an image to exactly target_w x target_h,
    since the image model doesn't guarantee an exact output size."""
    img = img.convert("RGB")
    w, h = img.size
    scale = max(target_w / w, target_h / h)
    new_w, new_h = int(w * scale) + 1, int(h * scale) + 1
    img = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
    x1 = (new_w - target_w) // 2
    y1 = (new_h - target_h) // 2
    return img.crop((x1, y1, x1 + target_w, y1 + target_h))


def generate_cartoon_image(prompt_text, index, target_w, target_h, seed):
    """Fetch one cartoon frame from Pollinations' free image endpoint (no
    key, no signup, no daily cap) at the given size. Falls back to a plain
    LIGHT frame if generation fails for any reason (timeout, bad response,
    etc.) so the pipeline never crashes and never shows a dark frame.
    Always paces itself ~15s before returning, success or fallback, since
    the anonymous tier is shared and this now gets called several times
    per scene."""
    fallback = PIL.Image.new("RGB", (target_w, target_h), (235, 240, 255))
    url = POLLINATIONS_URL.format(prompt=requests.utils.quote(prompt_text))
    params = {"width": target_w, "height": target_h, "seed": seed, "nologo": "true"}

    image = fallback
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image"):
                image = PIL.Image.open(io.BytesIO(resp.content))
                break
            print(f"Pollinations returned {resp.status_code} for scene {index}: {resp.text[:150]}")
        except Exception as e:
            print(f"[Attempt {attempt+1}/3] Image fetch failed for scene {index}: {type(e).__name__}: {e}")
        time.sleep(POLLINATIONS_PACING_SECONDS)

    time.sleep(POLLINATIONS_PACING_SECONDS)
    return image


def generate_scene_image(prompt_text, index, seed):
    """Fetch ONE oversized image (bigger than the final WIDTH x HEIGHT by
    PAN_MARGIN) and fit it to that oversized canvas, so build_ken_burns_clip()
    has real room to pan/zoom across it without ever hitting an edge."""
    src_w = int(WIDTH * (1 + PAN_MARGIN))
    src_h = int(HEIGHT * (1 + PAN_MARGIN))
    raw = generate_cartoon_image(prompt_text, index, src_w, src_h, seed)
    return fit_to_size(raw, src_w, src_h)


def _ken_burns_params(img_size):
    """Pick a random pan/zoom trajectory for one oversized image: which
    corner it drifts from/to, and whether it zooms in or out."""
    src_w, src_h = img_size
    max_dx = max(src_w - WIDTH, 1)
    max_dy = max(src_h - HEIGHT, 1)
    return {
        "zoom_in": random.random() < 0.5,
        "start": (random.uniform(0, max_dx), random.uniform(0, max_dy)),
        "end": (random.uniform(0, max_dx), random.uniform(0, max_dy)),
    }


def _ken_burns_frame(img, params, progress, zoom_ratio=KEN_BURNS_ZOOM_RATIO):
    """Render one frame of a continuous pan+zoom over `img` at the given
    progress (0..1 through however long this image is being shown)."""
    progress = min(max(progress, 0), 1)
    src_w, src_h = img.size
    zoom = 1 + zoom_ratio * (progress if params["zoom_in"] else (1 - progress))
    crop_w = min(src_w, WIDTH / zoom)
    crop_h = min(src_h, HEIGHT / zoom)
    (sx, sy), (ex, ey) = params["start"], params["end"]
    cx = sx + (ex - sx) * progress
    cy = sy + (ey - sy) * progress
    cx = max(0, min(cx, src_w - crop_w))
    cy = max(0, min(cy, src_h - crop_h))
    return img.crop((cx, cy, cx + crop_w, cy + crop_h)).resize((WIDTH, HEIGHT), PIL.Image.Resampling.LANCZOS)


def build_ken_burns_clip(img, duration):
    """Continuous pan + zoom over a single oversized still image."""
    params = _ken_burns_params(img.size)

    def make_frame(t):
        return np.array(_ken_burns_frame(img, params, t / duration))

    return VideoClip(make_frame, duration=duration)


def build_animated_scene_clip(images, duration):
    """Crossfade continuously through several distinct AI keyframes for
    this scene (each with its own slow pan+zoom), so the scene shows real
    story motion - the character/action visibly changing - instead of just
    the camera moving over one static picture."""
    if len(images) == 1:
        return build_ken_burns_clip(images[0], duration)

    n = len(images)
    seg_dur = duration / (n - 1)
    params = [_ken_burns_params(img.size) for img in images]

    def make_frame(t):
        t = min(max(t, 0), duration - 1e-6)
        seg = min(int(t / seg_dur), n - 2)
        local = (t - seg * seg_dur) / seg_dur
        blend = (1 - np.cos(np.pi * local)) / 2  # eased 0 -> 1 across the segment

        frame_a = _ken_burns_frame(images[seg], params[seg], (seg + local) / (n - 1))
        frame_b = _ken_burns_frame(images[seg + 1], params[seg + 1], (seg + 1 + local) / (n - 1))
        arr = np.array(frame_a).astype(np.float32) * (1 - blend) + np.array(frame_b).astype(np.float32) * blend
        return arr.astype("uint8")

    return VideoClip(make_frame, duration=duration)


def add_caption(bg_clip, caption_text, duration, is_cta=False, chunk_words=4):
    """Split the caption into short chunks that appear one after another in
    sync with the scene's audio - fast-paced style common on Shorts."""
    try:
        words = caption_text.split()
        chunks = [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), chunk_words)] or [caption_text]
        chunk_duration = duration / len(chunks)

        caption_clips = []
        for idx, chunk in enumerate(chunks):
            txt_clip = TextClip(
                font=FONT_PATH,
                text=chunk,
                font_size=72 if is_cta else 62,
                color="yellow" if is_cta else "white",
                stroke_color="black",
                stroke_width=3 if is_cta else 2,
                method="caption",
                size=(int(WIDTH * 0.85), None),
                text_align="center",
            ).with_duration(chunk_duration).with_start(idx * chunk_duration).with_position(("center", int(HEIGHT * 0.72)))
            caption_clips.append(txt_clip)

        return CompositeVideoClip([bg_clip] + caption_clips, size=(WIDTH, HEIGHT)).with_duration(duration)
    except Exception as e:
        print(f"Caption failed for text '{caption_text[:30]}...': {e}")
        return bg_clip


def add_title_flash(scene_clip, title_text, flash_duration=1.5):
    """Big bold title card for the first ~1.5s of the first scene, as a
    stronger scroll-stopping hook."""
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
    .mp3 files in a 'music' folder in the repo. If missing/empty, narration
    plays with no music."""
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
        music_quiet = music_full.with_volume_scaled(0.15)

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
            "categoryId": "1",  # Film & Animation
        },
        "status": {
            "privacyStatus": "public",
            # IMPORTANT: set this honestly based on the final content of
            # each video. If a story is genuinely made/targeted for
            # children, this must be True - YouTube/COPPA rules apply
            # regardless of what a script auto-generates.
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


def update_tracker(video_id, title, language):
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
        "language": language,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "url": f"https://youtube.com/shorts/{video_id}",
    })

    with open(TRACKER_FILE, "w") as f:
        json.dump(data, f, indent=2)


def cleanup_temp_files(num_scenes):
    for i in range(num_scenes):
        path = f"part_{i}.mp3"
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


def generate_video():
    target_duration = round(random.uniform(MIN_DURATION, MAX_DURATION), 1)
    language = get_run_config()

    trend_text = get_trending_inspiration()
    script = generate_script_with_ai(language, trend_text)
    title = script["title"]
    character_sheet = script["character_sheet"]
    cta_scene = CTA_SCENE_UR if language == "ur" else CTA_SCENE_EN
    scenes = list(script["scenes"]) + [cta_scene]
    tts_lang = "ur" if language == "ur" else "en"

    print(f"Language: {language} | Topic: {title} | Target duration: {target_duration}s")
    if trend_text:
        print(f"Trend inspiration used:\n{trend_text}")

    # Same base seed for every scene in this video, so the free image
    # endpoint stays in a consistent art style/palette across the story;
    # each beat still gets its own offset so the 3 frames within a scene
    # come out visibly different from each other.
    video_seed = random.randint(1, 999999)

    audio_clips = []
    video_clips = []
    for i, scene in enumerate(scenes):
        fname = f"part_{i}.mp3"
        gTTS(text=scene["text"], lang=tts_lang).save(fname)
        aclip = AudioFileClip(fname)
        audio_clips.append(aclip)

        beats = CTA_IMAGE_BEATS if scene.get("cta") else scene["image_prompts"]

        scene_images = []
        for b, beat in enumerate(beats):
            img_prompt = STYLE_PREFIX + "Character: " + character_sheet + ". Scene: " + beat
            beat_seed = video_seed + i * 10 + b
            scene_images.append(generate_scene_image(img_prompt, f"{i}.{b}", beat_seed))

        bg_clip = build_animated_scene_clip(scene_images, aclip.duration)

        caption_text = scene.get("caption") or scene.get("caption_roman") or scene["text"]
        scene_clip = add_caption(bg_clip, caption_text, aclip.duration, is_cta=scene.get("cta", False))
        if i == 0:
            scene_clip = add_title_flash(scene_clip, title)
        video_clips.append(scene_clip)

    final_audio = concatenate_audioclips(audio_clips)
    final_video = concatenate_videoclips(video_clips)

    current_duration = min(final_audio.duration, final_video.duration)

    if current_duration > target_duration:
        final_audio = final_audio.subclipped(0, target_duration)
        final_video = final_video.subclipped(0, target_duration)
    elif current_duration < target_duration:
        pad = target_duration - current_duration
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

    base_tags = ["cartoon", "shorts", "animation", "space"]
    ai_hashtags = [h.strip().lstrip("#").lower() for h in script.get("hashtags", []) if h.strip()]
    all_tags = list(dict.fromkeys(ai_hashtags + base_tags))

    hashtag_line = " ".join(f"#{t}" for t in all_tags[:8])
    subscribe_line = (
        "Rozana nayi cartoon video ke liye subscribe karein!" if language == "ur"
        else "Subscribe for a new cartoon every single day!"
    )
    description = f"{title}\n\n{subscribe_line}\n\n{hashtag_line}"

    video_id = upload_to_youtube(output_path, title, description, all_tags)
    update_tracker(video_id, title, language)
    cleanup_temp_files(len(scenes))


if __name__ == "__main__":
    generate_video()
