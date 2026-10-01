import os
import io
import json
import math
import bisect
import random
import time
import asyncio
import urllib.parse
from datetime import datetime, timezone

import requests
import numpy as np
import PIL.Image
import PIL.ImageOps

# Compatibility fix for Pillow & MoviePy
if not hasattr(PIL.Image, "ANTIALIAS"):
    PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS

import edge_tts
from moviepy import (
    AudioFileClip, ImageClip, VideoClip, concatenate_videoclips,
    concatenate_audioclips, TextClip, CompositeVideoClip, AudioClip,
    CompositeAudioClip, afx
)

from google import genai
from google.genai import types

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
WIDTH, HEIGHT = 1080, 1920
GEN_W, GEN_H = 720, 1280            # size requested from the image AI (faster, more reliable); upscaled to final
MIN_DURATION = 20.0                 # padded only if the natural length is shorter
MAX_DURATION = 40.0                 # trimmed only if the natural length is longer

GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.1-flash-lite"
POLLINATIONS_TOKEN = os.environ.get("POLLINATIONS_TOKEN")      # optional free token (fewer rate-limit errors)

EDGE_TTS_VOICE = os.environ.get("EDGE_TTS_VOICE", "en-US-AndrewNeural")
EDGE_TTS_RATE = os.environ.get("EDGE_TTS_RATE", "+12%")
EDGE_TTS_PITCH = os.environ.get("EDGE_TTS_PITCH", "+4Hz")      # slightly brighter = friendlier mascot voice

CAPTIONS_ENABLED = os.environ.get("CAPTIONS", "1") != "0"       # set CAPTIONS=0 to turn captions off
USE_CUTOUT = os.environ.get("USE_CUTOUT", "1") != "0"           # "talking" effect (needs rembg, optional)
REMBG_MODEL = os.environ.get("REMBG_MODEL", "u2netp")

TRACKER_FILE = "upload_tracker_cartoon.json"   # separate from the other channel's tracker
MUSIC_DIR = "music"
SFX_DIR = "sfx"

CATEGORY_IDS = {"ai_tips": "27", "random": "28"}
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

CAPTION_Y_RATIO = 0.48              # screen center: never hidden behind the Shorts UI
CAPTION_COLORS = ["white", "yellow"]

# Visual style presets. Change with VIDEO_STYLE=clay|anime (env var). Same mascot in every video = brand.
STYLE_PRESETS = {
    "clay": {
        "style": ("cute claymation style, handmade plasticine clay texture, soft fingerprints, "
                  "stop-motion look, soft studio lighting, vibrant colors, highly detailed"),
        "mascot": ("Bulby, a cute round clay lightbulb character with big friendly eyes, a glowing warm "
                   "yellow body, tiny arms and legs, and a cheerful smile"),
    },
    "anime": {
        "style": ("hand-painted dreamy anime film style, soft watercolor backgrounds, warm gentle light, "
                  "vibrant colors, highly detailed"),
        "mascot": ("Bulby, a cute round glowing lightbulb spirit with big friendly eyes, tiny arms and "
                   "legs, and a cheerful smile"),
    },
}
VIDEO_STYLE = os.environ.get("VIDEO_STYLE", "clay")
_PRESET = STYLE_PRESETS.get(VIDEO_STYLE, STYLE_PRESETS["clay"])
IMAGE_STYLE = _PRESET["style"] + ", vertical composition"
MASCOT = os.environ.get("MASCOT_DESC", _PRESET["mascot"])
IMAGE_NEGATIVE = "no text, no letters, no words, no watermark, no logo"

FRAMING = {
    "wide": "medium shot, the character centered and fully visible, simple uncluttered colorful background",
    "close": "close-up of the character's face and upper body, centered, expressive emotion, simple uncluttered background",
}

# Camera / motion tuning
ZOOM_AMT = 0.14        # extra zoom over the length of a shot
DRIFT = 40             # sideways drift in pixels
PUNCH_AMT = 0.025      # tiny "punch" zoom each time a new caption word appears
PUNCH_LEN = 0.14       # seconds
ENV_FPS = 30           # audio-loudness envelope resolution

RUN_SCHEDULE = {
    0: "ai_tips",
    4: "ai_tips",
    8: "ai_tips",
    12: "ai_tips",
    16: "ai_tips",
    20: "ai_tips",
}

LAST_GOOD_IMAGE = None
_CUTOUT_CACHE = {}
_REMBG_SESSION = None
_REMBG_FAILED = False


def get_run_config():
    hour = datetime.now(timezone.utc).hour
    closest_hour = min(RUN_SCHEDULE.keys(), key=lambda h: min(abs(hour - h), 24 - abs(hour - h)))
    return RUN_SCHEDULE[closest_hour]


CTA_SCENE = {
    "text": "Comment which AI trick you want next, and follow for more!",
    "queries": ["waving happily at the camera with a big smile, confetti in the air"],
    "cta": True,
}

FALLBACK_SCRIPTS = {
    "ai_tips": {
        "title": "You're Using ChatGPT Wrong",
        "hashtags": ["ai", "aitips", "chatgpt", "animation", "productivity"],
        "scenes": [
            {"text": "You're using AI chatbots wrong, and it costs you better answers.",
             "queries": ["sitting at a glowing computer looking confused in a cozy room",
                         "confused face with a question mark above its head"]},
            {"text": "Here's the fix: show the AI an example of what you want first.",
             "queries": ["holding a glowing example card next to a friendly robot in a bright classroom",
                         "excited face, eyes wide, pointing upward"]},
            {"text": "This is called few shot prompting, and AI learns faster from examples.",
             "queries": ["pointing at a colorful chalkboard with simple shapes in a classroom",
                         "proud smiling face with sparkles around"]},
            {"text": "Try it in your next chat, and watch the answers get better.",
             "queries": ["holding a futuristic phone with glowing bright light in a neon city",
                         "happy surprised face with shining eyes"]},
        ],
    },
}


# ----------------------------------------------------------------------------
# Tracker / topics
# ----------------------------------------------------------------------------

def _read_tracker():
    if not os.path.exists(TRACKER_FILE):
        return []
    try:
        with open(TRACKER_FILE) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def get_past_titles(limit=100):
    return [e.get("title", "") for e in _read_tracker()[-limit:] if e.get("title")]


AI_SUBTOPICS = [
    "writing better prompts (prompt techniques and formulas)",
    "using AI to boost productivity and save time at work",
    "using AI for studying, learning faster, and exam prep",
    "using AI for coding and learning to program as a beginner",
    "free AI tools most people don't know about",
    "common mistakes beginners make with AI chatbots",
    "how AI actually works, explained simply (no jargon)",
    "using AI for writing: emails, resumes, and social media posts",
    "using AI for research, summarizing, and fact-checking",
    "using AI for side income, freelancing, and career growth",
    "AI for creativity: ideas, design, video, and images",
    "AI safety and privacy tips everyday users should know",
]


def pick_subtopic():
    history = [e.get("subtopic") for e in _read_tracker() if e.get("subtopic")]
    last_used = {sub: idx for idx, sub in enumerate(history)}
    candidates = sorted(AI_SUBTOPICS, key=lambda t: (last_used.get(t, -1), random.random()))
    return candidates[0]


# ----------------------------------------------------------------------------
# Script generation (Gemini)
# ----------------------------------------------------------------------------

def build_prompt(content_type, avoid_text, subtopic=None):
    focus = f"Focus on: {subtopic}" if subtopic else ""
    return f"""You write short, punchy scripts for a YouTube Shorts channel where Bulby, a cute talking lightbulb
mascot, teaches everyday people (kids, teens and adults) practical AI tips. Bulby speaks directly to the
viewer in a friendly, energetic, slightly funny voice. No jargon.
{focus}
{avoid_text}

SCRIPT RULES:
- HOOK: the first sentence is a bold, surprising CLAIM or STATEMENT of max 12 words that makes people NEED to hear the rest (e.g. "You're using ChatGPT wrong, and it's costing you hours."). NEVER open with a question. NEVER start with "Did you know", "Have you ever", "What if", "Imagine", "Hi", "Hey", or any greeting/intro.
- 4 to 5 scenes. Each scene is ONE short sentence of 6 to 14 words (short sentences = fast cuts = better retention).
- About 45-60 words in total. Teach ONE concrete, useful tip and include one specific, copy-able example (an exact phrase the viewer can try).
- The LAST sentence should connect back to the hook so the video loops naturally when replayed.
- Do NOT add a subscribe/follow call-to-action; it is added automatically.

IMAGE RULES (each scene shows Bulby in 2 AI-generated shots):
- "queries" has EXACTLY 2 items per scene: [1] a medium shot of what Bulby is doing + the setting, [2] a close-up of Bulby's face showing a clear emotion that matches the sentence.
- NEVER describe Bulby's appearance (it is added automatically). Describe only actions, setting, mood and props.
- Make every scene a different place and color mood. Never ask for text, letters, logos, or screens with readable writing. Each query under 25 words.

Return ONLY valid JSON in exactly this shape:
{{
  "title": "curiosity-driven title under 8 words",
  "hashtags": ["5 to 8 relevant lowercase hashtags, no # symbol"],
  "scenes": [
    {{"text": "one short spoken sentence", "queries": ["medium shot: action + setting", "close-up: face + emotion"]}}
  ]
}}
"""


def validate_script(data):
    if not isinstance(data, dict) or not data.get("title") or not data.get("scenes"):
        raise ValueError("JSON missing 'title' or 'scenes'")
    for scene in data["scenes"]:
        if not scene.get("text"):
            raise ValueError("A scene is missing 'text'")
        queries = scene.get("queries")
        if isinstance(queries, str):
            queries = [queries]
        if not queries and scene.get("query"):
            queries = [scene["query"]]
        queries = [q for q in (queries or []) if isinstance(q, str) and q.strip()]
        if not queries:
            raise ValueError("A scene is missing image 'queries'")
        scene["queries"] = queries[:2]
    first = data["scenes"][0]["text"].strip()
    if first.endswith("?") or first.lower().startswith(("did you know", "have you ever", "what if", "imagine", "hi", "hey")):
        raise ValueError(f"Weak hook: {first[:60]}")
    data.setdefault("hashtags", [])


def generate_script_with_ai(content_type, subtopic=None, max_attempts=3):
    if not GEMINI_KEY:
        print("GEMINI_API_KEY not set, using fallback script.")
        return FALLBACK_SCRIPTS["ai_tips"]

    past_titles = get_past_titles()
    avoid_text = f"Do NOT repeat these topics: {'; '.join(past_titles)}" if past_titles else ""
    prompt = build_prompt(content_type, avoid_text, subtopic)

    try:
        client = genai.Client(api_key=GEMINI_KEY)
    except Exception as e:
        print(f"Gemini client error, using fallback: {e}")
        return FALLBACK_SCRIPTS["ai_tips"]

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            raw_text = getattr(response, "text", None)
            if not raw_text:
                raise ValueError("Empty response")

            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].strip()

            data = json.loads(cleaned)
            validate_script(data)
            print(f"Gemini script OK on attempt {attempt}/{max_attempts}.")
            return data
        except Exception as e:
            print(f"[Gemini attempt {attempt}/{max_attempts}] {type(e).__name__}: {e}")
            time.sleep(2 * attempt)

    print("All Gemini attempts failed, using fallback script.")
    return FALLBACK_SCRIPTS["ai_tips"]


# ----------------------------------------------------------------------------
# Voice (edge-tts) with word timings for synced captions
# ----------------------------------------------------------------------------

async def _edge_tts_save(text, voice, output_path):
    # edge-tts >= 7 defaults to sentence-level timings; word-level must be requested explicitly.
    kwargs = {"rate": EDGE_TTS_RATE, "pitch": EDGE_TTS_PITCH}
    try:
        communicate = edge_tts.Communicate(text, voice, boundary="WordBoundary", **kwargs)
    except TypeError:
        communicate = edge_tts.Communicate(text, voice, **kwargs)

    word_timings = []
    with open(output_path, "wb") as f:
        async for chunk in communicate.stream():
            ctype = chunk.get("type")
            if ctype == "audio":
                f.write(chunk["data"])
            elif ctype == "WordBoundary":
                try:
                    start = chunk["offset"] / 1e7
                    dur = chunk["duration"] / 1e7
                    word_timings.append({"text": chunk["text"], "start": start, "end": start + dur})
                except Exception:
                    pass
    return word_timings


def synthesize_speech(text, output_path, attempts=3):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            timings = asyncio.run(_edge_tts_save(text, EDGE_TTS_VOICE, output_path))
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return timings
            raise ValueError("empty audio file")
        except Exception as e:
            last_error = e
            print(f"[TTS attempt {attempt}/{attempts}] {type(e).__name__}: {e}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"edge-tts failed after {attempts} attempts: {last_error}")


def audio_envelope(aclip, fps=ENV_FPS):
    """Loudness per video frame (0..1) used to make the mascot 'talk'."""
    try:
        sr = 16000
        arr = aclip.to_soundarray(fps=sr)
        mono = np.abs(arr).mean(axis=1) if arr.ndim == 2 else np.abs(arr)
        step = sr // fps
        n = len(mono) // step
        if n < 2:
            return None
        env = mono[:n * step].reshape(n, step).mean(axis=1)
        ref = float(np.percentile(env, 95))
        if ref <= 1e-6:
            return None
        env = np.clip(env / ref, 0.0, 1.0)
        return np.convolve(env, np.ones(3) / 3, mode="same")
    except Exception as e:
        print(f"Audio envelope failed: {e}")
        return None


# ----------------------------------------------------------------------------
# Images (Pollinations, free)
# ----------------------------------------------------------------------------

def make_gradient_image(path):
    top = np.array(random.choice([(88, 40, 200), (20, 120, 220), (220, 70, 140), (30, 170, 150)]), dtype=float)
    bottom = np.array((15, 15, 40), dtype=float)
    ratio = np.linspace(0, 1, HEIGHT)[:, None, None]
    arr = top * (1 - ratio) + bottom * ratio
    arr = np.repeat(arr, WIDTH, axis=1).astype(np.uint8)
    PIL.Image.fromarray(arr).save(path, quality=95)


def generate_cartoon_image(prompt, out_path, seed, attempts=4):
    """Free image generation via Pollinations with retries, model switch and validation."""
    full_prompt = f"{IMAGE_STYLE}, {prompt}, {IMAGE_NEGATIVE}"[:900]
    encoded = urllib.parse.quote(full_prompt)
    headers = {"User-Agent": "Mozilla/5.0"}
    if POLLINATIONS_TOKEN:
        headers["Authorization"] = f"Bearer {POLLINATIONS_TOKEN}"

    for attempt in range(1, attempts + 1):
        model = "flux" if attempt <= 2 else "turbo"
        url = (f"https://image.pollinations.ai/prompt/{encoded}"
               f"?width={GEN_W}&height={GEN_H}&seed={seed}&model={model}&nologo=true&enhance=false")
        try:
            resp = requests.get(url, headers=headers, timeout=120)
            if resp.status_code == 200 and resp.content:
                img = PIL.Image.open(io.BytesIO(resp.content)).convert("RGB")
                if min(img.size) < 256:
                    raise ValueError(f"image too small: {img.size}")
                img = PIL.ImageOps.fit(img, (WIDTH, HEIGHT), method=PIL.Image.Resampling.LANCZOS)
                img.save(out_path, quality=95)
                return True
            print(f"[Image attempt {attempt}/{attempts}] status {resp.status_code}")
        except Exception as e:
            print(f"[Image attempt {attempt}/{attempts}] {type(e).__name__}: {e}")
        time.sleep(4 * attempt)
    return False


def get_scene_image(query, kind, index, shot, base_seed):
    """Returns the path of a usable image for this shot (never raises)."""
    global LAST_GOOD_IMAGE
    path = f"cartoon_{index}_{shot}.jpg"
    prompt = f"{MASCOT}, {query}, {FRAMING[kind]}"
    if generate_cartoon_image(prompt, path, seed=base_seed + index * 10 + shot):
        LAST_GOOD_IMAGE = path
        time.sleep(1.5)  # be gentle with the free API
        return path
    if LAST_GOOD_IMAGE:
        print(f"Scene {index} shot {shot}: image failed, reusing previous image.")
        return LAST_GOOD_IMAGE
    print(f"Scene {index} shot {shot}: image failed, using gradient background.")
    make_gradient_image(path)
    return path


# ----------------------------------------------------------------------------
# "Talking" cutout (optional, needs rembg) + camera + shots
# ----------------------------------------------------------------------------

def get_cutout(img_path):
    """Returns (RGBA cutout of the mascot, (pivot_x, pivot_y)) or None. Fully optional."""
    global _REMBG_SESSION, _REMBG_FAILED
    if not USE_CUTOUT or _REMBG_FAILED:
        return None
    if img_path in _CUTOUT_CACHE:
        return _CUTOUT_CACHE[img_path]

    result = None
    try:
        import rembg
        if _REMBG_SESSION is None:
            _REMBG_SESSION = rembg.new_session(REMBG_MODEL)
        img = PIL.Image.open(img_path).convert("RGB")
        out = rembg.remove(img, session=_REMBG_SESSION).convert("RGBA")
        alpha = np.asarray(out.split()[-1])
        coverage = float((alpha > 128).mean())
        bbox = out.getbbox()
        if bbox and 0.04 < coverage < 0.80:
            result = (out, ((bbox[0] + bbox[2]) / 2.0, float(bbox[3])))
        else:
            print(f"Cutout rejected (coverage={coverage:.2f}), using whole-frame pulse.")
    except ImportError:
        print("rembg not installed: using whole-frame pulse instead of cutout.")
        _REMBG_FAILED = True
    except Exception as e:
        print(f"Cutout failed ({type(e).__name__}: {e}), using whole-frame pulse.")
    _CUTOUT_CACHE[img_path] = result
    return result


def _affine_const():
    return PIL.Image.Transform.AFFINE if hasattr(PIL.Image, "Transform") else PIL.Image.AFFINE


def _bilinear():
    return PIL.Image.Resampling.BILINEAR if hasattr(PIL.Image, "Resampling") else PIL.Image.BILINEAR


def camera_affine(s, cx, cy):
    """Affine data for a window of size (W/s, H/s) centered at (cx, cy), clamped inside the image."""
    half_w, half_h = WIDTH / (2 * s), HEIGHT / (2 * s)
    cx = min(max(cx, half_w), WIDTH - half_w)
    cy = min(max(cy, half_h), HEIGHT - half_h)
    return (1 / s, 0, cx - WIDTH / (2 * s), 0, 1 / s, cy - HEIGHT / (2 * s))


def _env_value(env, tt):
    if env is None or len(env) == 0:
        return 0.0
    idx = min(max(int(tt * ENV_FPS), 0), len(env) - 1)
    return float(env[idx])


def _punch_value(times, tt):
    if not times:
        return 0.0
    i = bisect.bisect_right(times, tt) - 1
    if i < 0:
        return 0.0
    return PUNCH_AMT * max(0.0, 1.0 - (tt - times[i]) / PUNCH_LEN)


def make_shot_clip(img_path, duration, zoom_in, base_zoom, focus_y, env, env_offset, punch_times):
    """One camera shot: eased zoom + drift + punch zoom on each caption word, and the mascot
    squashes/stretches/bounces with the loudness of the voice so it looks like it is talking."""
    bg = PIL.Image.open(img_path).convert("RGBA")
    if bg.size != (WIDTH, HEIGHT):
        bg = bg.resize((WIDTH, HEIGHT), _bilinear())
    cut = get_cutout(img_path)
    direction = random.choice([-1, 1])
    affine, bilinear = _affine_const(), _bilinear()

    def progress(t):
        p = min(max(t / duration, 0.0), 1.0)
        return p * p * (3 - 2 * p)

    def frame(t):
        tt = env_offset + t
        e = _env_value(env, tt)
        p = progress(t)
        s = base_zoom * (1.0 + ZOOM_AMT * (p if zoom_in else 1.0 - p))
        s *= 1.0 + _punch_value(punch_times, tt)

        if cut is None:
            s *= 1.0 + 0.012 * e
            comp = bg
        else:
            cut_img, (px, py) = cut
            lift = 16.0 * e + 2.5 * math.sin(tt * 7.5)
            sy = 1.0 + 0.05 * e
            sx = 1.0 - 0.02 * e
            data = (1 / sx, 0, px - px / sx, 0, 1 / sy, py + (lift - py) / sy)
            fg = cut_img.transform((WIDTH, HEIGHT), affine, data, resample=bilinear)
            comp = PIL.Image.alpha_composite(bg, fg)

        cx = WIDTH / 2 + direction * DRIFT * (p - 0.5)
        cy = focus_y * HEIGHT
        img = comp.transform((WIDTH, HEIGHT), affine, camera_affine(s, cx, cy), resample=bilinear)
        return np.asarray(img.convert("RGB"))

    try:
        frame(0.0)  # smoke test
        return VideoClip(frame_function=frame, duration=duration)
    except Exception as e:
        print(f"Shot rendering setup failed ({type(e).__name__}: {e}), using static image.")
        return ImageClip(np.asarray(bg.convert("RGB"))).with_duration(duration)


def build_scene_background(scene, duration, index, base_seed, env, punch_times):
    queries = scene["queries"]
    n_shots = 1 if (scene.get("cta") or duration < 2.8) else 2
    shots, prev_path, elapsed = [], None, 0.0

    for k in range(n_shots):
        shot_dur = duration / n_shots if k < n_shots - 1 else duration - elapsed
        same_image = k >= len(queries)  # only one query given: reuse the image as a close-up crop
        if same_image and prev_path:
            path, base_zoom, focus_y = prev_path, 1.35, 0.40
        else:
            kind = "wide" if k == 0 else "close"
            path = get_scene_image(queries[min(k, len(queries) - 1)], kind, index, k, base_seed)
            base_zoom, focus_y = 1.04, 0.5
        prev_path = path
        shots.append(make_shot_clip(
            path, shot_dur, zoom_in=((index + k) % 2 == 0), base_zoom=base_zoom, focus_y=focus_y,
            env=env, env_offset=elapsed, punch_times=punch_times,
        ))
        elapsed += shot_dur

    return shots[0] if len(shots) == 1 else concatenate_videoclips(shots)


# ----------------------------------------------------------------------------
# Captions / title
# ----------------------------------------------------------------------------

def _text_clip(**kwargs):
    kwargs.setdefault("font", FONT_PATH if os.path.exists(FONT_PATH) else None)
    try:
        return TextClip(margin=(30, 30), **kwargs)
    except TypeError:
        return TextClip(**kwargs)


def build_caption_segments(caption_text, word_timings, duration, max_words=2, max_chars=14):
    """[(TEXT, start, end)] in 1-2 word chunks. Uses real word timings when available."""
    segments = []
    if word_timings:
        groups, cur = [], []
        for w in word_timings:
            cand = cur + [w]
            if cur and (len(cand) > max_words or len(" ".join(x["text"] for x in cand)) > max_chars):
                groups.append(cur)
                cur = [w]
            else:
                cur = cand
        if cur:
            groups.append(cur)
        starts = [0.0] + [min(g[0]["start"], duration) for g in groups[1:]]
        for i, g in enumerate(groups):
            end = starts[i + 1] if i + 1 < len(groups) else duration
            if end - starts[i] > 0.05:
                segments.append((" ".join(x["text"] for x in g).upper(), starts[i], end))
    else:
        chunks, cur = [], []
        for word in caption_text.split():
            cand = cur + [word]
            if cur and (len(cand) > max_words or len(" ".join(cand)) > max_chars):
                chunks.append(" ".join(cur))
                cur = [word]
            else:
                cur = cand
        if cur:
            chunks.append(" ".join(cur))
        weights = [len(c) + 2 for c in chunks] or [1]
        total, start = sum(weights), 0.0
        for c, wgt in zip(chunks, weights):
            seg = duration * wgt / total
            segments.append((c.upper(), start, start + seg))
            start += seg
    return segments


def add_caption(bg_clip, segments, duration, is_cta=False):
    try:
        caption_clips = []
        for idx, (text, start, end) in enumerate(segments):
            txt = _text_clip(
                text=text,
                font_size=88 if is_cta else 84,
                color="yellow" if is_cta else CAPTION_COLORS[idx % len(CAPTION_COLORS)],
                stroke_color="black",
                stroke_width=7,
                method="caption",
                size=(int(WIDTH * 0.88), None),
                text_align="center",
            )
            y = int(HEIGHT * CAPTION_Y_RATIO - txt.h / 2)
            caption_clips.append(txt.with_duration(end - start).with_start(start).with_position(("center", y)))
        return CompositeVideoClip([bg_clip] + caption_clips, size=(WIDTH, HEIGHT)).with_duration(duration)
    except Exception as e:
        print(f"Caption rendering failed: {e}")
        return bg_clip


def add_title_flash(scene_clip, title_text, flash_duration=1.8):
    try:
        flash_len = min(flash_duration, scene_clip.duration)
        title_clip = _text_clip(
            text=title_text.upper(),
            font_size=70,
            color="white",
            stroke_color="black",
            stroke_width=6,
            method="caption",
            size=(int(WIDTH * 0.88), None),
            text_align="center",
        )
        y = int(HEIGHT * 0.20 - title_clip.h / 2)
        title_clip = title_clip.with_duration(flash_len).with_position(("center", y))
        return CompositeVideoClip([scene_clip, title_clip], size=(WIDTH, HEIGHT)).with_duration(scene_clip.duration)
    except Exception as e:
        print(f"Title flash failed: {e}")
        return scene_clip


# ----------------------------------------------------------------------------
# Audio extras
# ----------------------------------------------------------------------------

def add_background_music(narration_audio, duration):
    if not os.path.isdir(MUSIC_DIR):
        return narration_audio
    tracks = [f for f in os.listdir(MUSIC_DIR) if f.lower().endswith((".mp3", ".wav", ".m4a"))]
    if not tracks:
        return narration_audio
    try:
        music = AudioFileClip(os.path.join(MUSIC_DIR, random.choice(tracks)))
        loop_clips, cur_dur = [], 0
        while cur_dur < duration:
            loop_clips.append(music)
            cur_dur += music.duration
        music_full = concatenate_audioclips(loop_clips).subclipped(0, duration)
        return CompositeAudioClip([narration_audio, music_full.with_volume_scaled(0.10)])
    except Exception as e:
        print(f"Background music error: {e}")
        return narration_audio


def add_sfx_transitions(audio, transition_times, duration):
    if not os.path.isdir(SFX_DIR):
        return audio
    files = [f for f in os.listdir(SFX_DIR) if f.lower().endswith((".mp3", ".wav", ".m4a"))]
    if not files:
        return audio
    preferred = [f for f in files if f.lower().startswith("whoosh")]
    sfx_path = os.path.join(SFX_DIR, random.choice(preferred or files))
    try:
        layers = [audio]
        for t in transition_times:
            if 0 < t < duration:
                sfx = AudioFileClip(sfx_path)
                sfx = sfx.subclipped(0, min(1.0, sfx.duration)).with_volume_scaled(0.45)
                layers.append(sfx.with_start(max(t - 0.12, 0)))
        return CompositeAudioClip(layers)
    except Exception as e:
        print(f"SFX error: {e}")
        return audio


def silence_clip(duration):
    def frame(t):
        t = np.asarray(t)
        return np.zeros((*t.shape, 2))
    return AudioClip(frame, duration=duration, fps=44100)


# ----------------------------------------------------------------------------
# YouTube
# ----------------------------------------------------------------------------

def get_youtube_client():
    token_raw = os.environ.get("TOKEN_JSON")
    client_raw = os.environ.get("CLIENT_SECRET_JSON")

    if not token_raw or not client_raw:
        raise RuntimeError("Missing TOKEN_JSON or CLIENT_SECRET_JSON secret.")

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
            raise RuntimeError("Invalid TOKEN_JSON: no refresh_token available.")

    return build("youtube", "v3", credentials=creds)


def upload_to_youtube(video_path, title, description, tags, category_id="27"):
    youtube = get_youtube_client()
    body = {
        "snippet": {"title": title[:100], "description": description, "tags": tags, "categoryId": category_id},
        "status": {"privacyStatus": "public", "selfDeclaredMadeForKids": False},
    }

    media = MediaFileUpload(video_path, chunksize=-1, resumable=True, mimetype="video/mp4")
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"Upload progress: {int(status.progress() * 100)}%")

    print(f"Uploaded: https://youtube.com/shorts/{response['id']}")
    return response["id"]


def update_tracker(video_id, title, content_type, subtopic=None):
    data = _read_tracker()
    data.append({
        "video_id": video_id,
        "title": title,
        "content_type": content_type,
        "subtopic": subtopic,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "url": f"https://youtube.com/shorts/{video_id}",
    })
    with open(TRACKER_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------

def generate_video():
    content_type = get_run_config()
    subtopic = pick_subtopic() if content_type == "ai_tips" else None

    script = generate_script_with_ai(content_type, subtopic)
    title = script["title"]
    scenes = list(script["scenes"]) + [CTA_SCENE]
    base_seed = random.randint(1, 900000)

    print(f"Type: {content_type} | Style: {VIDEO_STYLE} | Subtopic: {subtopic} | Title: {title}")

    audio_clips, video_clips = [], []
    for i, scene in enumerate(scenes):
        fname = f"part_{i}.mp3"
        word_timings = synthesize_speech(scene["text"], fname)
        aclip = AudioFileClip(fname)
        audio_clips.append(aclip)
        duration = aclip.duration

        env = audio_envelope(aclip)
        segments = build_caption_segments(scene["text"], word_timings, duration)
        punch_times = [s[1] for s in segments]

        bg_clip = build_scene_background(scene, duration, i, base_seed, env, punch_times)
        scene_clip = add_caption(bg_clip, segments, duration, is_cta=scene.get("cta", False)) if CAPTIONS_ENABLED else bg_clip
        if i == 0:
            scene_clip = add_title_flash(scene_clip, title)
        video_clips.append(scene_clip)

    transition_times, t = [], 0.0
    for a in audio_clips[:-1]:
        t += a.duration
        transition_times.append(t)

    final_audio = concatenate_audioclips(audio_clips)
    final_video = concatenate_videoclips(video_clips)
    current_duration = min(final_audio.duration, final_video.duration)

    # Keep the natural length (no mid-sentence cuts); only enforce the limits.
    if current_duration > MAX_DURATION:
        final_audio = final_audio.subclipped(0, MAX_DURATION)
        final_video = final_video.subclipped(0, MAX_DURATION)
    elif current_duration < MIN_DURATION:
        pad = MIN_DURATION - current_duration
        final_audio = concatenate_audioclips([final_audio, silence_clip(pad)])
        last_frame = final_video.get_frame(max(final_video.duration - 0.04, 0))
        final_video = concatenate_videoclips([final_video, ImageClip(last_frame).with_duration(pad)])

    final_duration = min(final_audio.duration, final_video.duration)
    final_audio = final_audio.subclipped(0, final_duration)
    final_video = final_video.subclipped(0, final_duration)

    final_audio = add_background_music(final_audio, final_duration)
    final_audio = add_sfx_transitions(final_audio, transition_times, final_duration)
    final_audio = final_audio.with_duration(final_duration)
    try:
        final_audio = final_audio.with_effects([afx.AudioFadeOut(0.4)])
    except Exception as e:
        print(f"Audio fade-out skipped: {e}")

    final = final_video.with_audio(final_audio)
    output_path = "final_short_cartoon.mp4"
    final.write_videofile(
        output_path, fps=30, codec="libx264", audio_codec="aac",
        bitrate="6000k", preset="veryfast", threads=4,
    )

    base_tags = ["ai", "aitips", "chatgpt", "animation", "shorts"]
    ai_hashtags = [h.strip().lstrip("#").lower() for h in script.get("hashtags", []) if h.strip()]
    all_tags = list(dict.fromkeys(ai_hashtags + base_tags))

    hashtag_line = " ".join(f"#{t}" for t in all_tags[:8])
    description = f"{title}\n\nSubscribe for a new AI trick every single day!\n\n{hashtag_line}"

    video_id = upload_to_youtube(
        output_path, title, description, all_tags,
        category_id=CATEGORY_IDS.get(content_type, "27"),
    )
    update_tracker(video_id, title, content_type, subtopic)


if __name__ == "__main__":
    generate_video()
