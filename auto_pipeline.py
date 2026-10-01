import os
import json
import random
import time
import asyncio
from datetime import datetime, timezone

import requests
import numpy as np
import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"): PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS
import edge_tts
from moviepy import (
    VideoFileClip, AudioFileClip, ImageClip, concatenate_videoclips,
    concatenate_audioclips, TextClip, CompositeVideoClip, AudioClip,
    CompositeAudioClip, vfx
)

from google import genai
from google.genai import types

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

WIDTH, HEIGHT = 1080, 1920
MIN_DURATION = 18.0
MAX_DURATION = 28.0  # final video is trimmed to MAX / padded to MIN only if the natural length falls outside this range
PEXELS_KEY = os.environ.get("PEXELS_API_KEY")
PIXABAY_KEY = os.environ.get("PIXABAY_API_KEY")  # optional, free: https://pixabay.com/api/docs/
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.1-flash-lite"  # cheap + fast, plenty for short scripts

TRACKER_FILE = "upload_tracker.json"
MUSIC_DIR = "music"  # royalty-free .mp3 files; one is picked at random each run
SFX_DIR = "sfx"      # short transition sounds (e.g. whoosh.mp3); optional

# Pacing: narration is sped up via edge-tts' native rate (keeps pitch natural),
# and stock footage is sped up with MoviePy's MultiplySpeed so visuals feel snappy too.
SPEED_FACTOR = 1.12
NARRATION_RATE = f"+{int(round((SPEED_FACTOR - 1) * 100))}%"

VOICES = {
    "en": "en-US-ChristopherNeural",
    "ur": "ur-PK-AsadNeural",
}

# Font used for on-screen captions (installed via apt in the workflow: fonts-dejavu-core).
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Captions sit at screen center so the Shorts UI (title, buttons, channel) never covers them.
CAPTION_Y_RATIO = 0.48
CAPTION_COLORS = ["white", "yellow"]  # alternates per word chunk for a Hormozi-style pop

# 6 runs per day, one every 4 hours (UTC), split 3 English / 3 Urdu and evenly
# across both content types. Workflow cron: '0 */4 * * *'.
RUN_SCHEDULE = {
    0: ("random", "en"),
    4: ("ai_tips", "ur"),
    8: ("random", "ur"),
    12: ("ai_tips", "en"),
    16: ("random", "en"),
    20: ("ai_tips", "ur"),
}

USED_CLIP_IDS = set()  # avoid the same stock clip appearing twice in one video


def get_run_config():
    """Decide today's content type + language from the UTC hour. Snaps to the
    nearest scheduled hour so a slightly delayed workflow run (common with
    GitHub Actions cron) still picks a sensible slot instead of crashing."""
    hour = datetime.now(timezone.utc).hour
    closest_hour = min(RUN_SCHEDULE.keys(), key=lambda h: min(abs(hour - h), 24 - abs(hour - h)))
    return RUN_SCHEDULE[closest_hour]


# Urdu runs: "text" stays in proper Urdu script because TTS needs real Urdu
# script to pronounce it correctly - that field is only ever used for the
# voice-over, never shown on screen. "caption" is Roman Urdu.
CTA_SCENE_EN = {
    "text": "Follow for a brand new one every single day.",
    "query": "neon gradient abstract motion background",
    "cta": True,
    "caption": "FOLLOW FOR A NEW ONE EVERY DAY!",
}

CTA_SCENE_UR = {
    "text": "روزانہ نئی ویڈیو کے لیے فالو کریں۔",
    "query": "neon gradient abstract motion background",
    "cta": True,
    "caption": "ROZANA NAYI VIDEO KE LIYE FOLLOW KAREIN!",
}

# Used only if AI script generation fails, so the pipeline never crashes.
FALLBACK_SCRIPTS = {
    "random": {
        "en": {
            "title": "Space is Completely Silent",
            "hashtags": ["space", "facts", "didyouknow", "science", "universe"],
            "scenes": [
                {"text": "Did you know that space is completely silent?", "query": "deep space cosmos silence"},
                {"text": "Sound waves require a medium like air to travel, and because space is a vacuum, molecules are too far apart to carry sound.", "query": "space vacuum stars"},
                {"text": "If you screamed in space, no one would hear you.", "query": "astronaut floating space"},
                {"text": "This eerie silence stretches across the entire universe, making the cosmos both breathtaking and strangely terrifying.", "query": "galaxy spinning nebula"},
            ],
        },
        "ur": {
            "title": "Space Mein Mukammal Khamoshi Hoti Hai",
            "hashtags": ["space", "facts", "didyouknow", "science", "universe"],
            "scenes": [
                {"text": "کیا آپ جانتے ہیں کہ خلا میں مکمل خاموشی ہوتی ہے؟", "caption_roman": "Kya aap jantay hain ke khala mein mukammal khamoshi hoti hai?", "query": "deep space cosmos silence"},
                {"text": "آواز کی لہروں کو سفر کے لیے ہوا جیسے میڈیم کی ضرورت ہوتی ہے، اور چونکہ خلا ایک خلا ہے، اس لیے ذرات ایک دوسرے سے بہت دور ہوتے ہیں۔", "caption_roman": "Awaz ki lehron ko safar ke liye hawa jaisay medium ki zaroorat hoti hai, aur chunke khala aik khala hai, is liye zarrat aik dosray se bohat door hotay hain.", "query": "space vacuum stars"},
                {"text": "اگر آپ خلا میں چیخیں تو کوئی نہیں سنے گا۔", "caption_roman": "Agar aap khala mein cheekhein to koi nahi sunega.", "query": "astronaut floating space"},
                {"text": "یہ پراسرار خاموشی پوری کائنات میں پھیلی ہوئی ہے، جو کائنات کو خوبصورت اور عجیب طور پر خوفناک دونوں بناتی ہے۔", "caption_roman": "Ye pur-israr khamoshi puri kayenat mein phaili hui hai, jo kayenat ko khoobsurat aur ajeeb tarah se khaufnak dono banati hai.", "query": "galaxy spinning nebula"},
            ],
        },
    },
    "ai_tips": {
        "en": {
            "title": "This One Prompt Trick Changes Everything",
            "hashtags": ["ai", "aitips", "chatgpt", "prompting", "productivity"],
            "scenes": [
                {"text": "Most people use AI chatbots completely wrong, and it's costing them much better answers.", "query": "person typing laptop screen"},
                {"text": "Here's a simple trick: instead of just asking a question, show the AI an example of what you want first.", "query": "hands typing keyboard closeup"},
                {"text": "This is called few shot prompting, and it works because AI learns better from examples than from instructions alone.", "query": "digital technology abstract lights"},
                {"text": "Try this in your very next chat with any AI tool, and watch how much better the answer gets.", "query": "smartphone chat app screen"},
            ],
        },
        "ur": {
            "title": "Ye Aik Trick AI Istemal Karne Ka Tareeqa Badal Degi",
            "hashtags": ["ai", "aitips", "chatgpt", "prompting", "productivity"],
            "scenes": [
                {"text": "زیادہ تر لوگ AI چیٹ بوٹس کا غلط استعمال کرتے ہیں، اور اس کی وجہ سے انہیں بہتر جوابات نہیں ملتے۔", "caption_roman": "Ziyada tar log AI chatbots ka ghalat istemal kartay hain, aur is ki wajah se unhein behtar jawabat nahi miltay.", "query": "person typing laptop screen"},
                {"text": "یہ آسان ترکیب آزمائیں: صرف سوال پوچھنے کے بجائے، پہلے AI کو ایک مثال دکھائیں کہ آپ کیا چاہتے ہیں۔", "caption_roman": "Ye aasan trick azmayein: sirf sawaal poochne ke bajaye, pehlay AI ko aik misaal dikhayein ke aap kya chahtay hain.", "query": "hands typing keyboard closeup"},
                {"text": "اسے فیو شاٹ پرامپٹنگ کہتے ہیں، اور یہ اس لیے کام کرتا ہے کیونکہ AI مثالوں سے ہدایات کی نسبت بہتر سیکھتا ہے۔", "caption_roman": "Isay few-shot prompting kehtay hain, aur ye is liye kaam karta hai kyunke AI misaalon se hidayaat ki nisbat behtar seekhta hai.", "query": "digital technology abstract lights"},
                {"text": "اپنی اگلی چیٹ میں یہ ضرور آزمائیں اور دیکھیں جواب کتنا بہتر ملتا ہے۔", "caption_roman": "Apni agli chat mein ye zaroor azmayein aur dekhein jawab kitna behtar milta hai.", "query": "smartphone chat app screen"},
            ],
        },
    },
}


# ----------------------------------------------------------------------------
# Script generation (Gemini)
# ----------------------------------------------------------------------------

def get_past_titles(language, limit=20):
    """Titles of previously uploaded videos in the same language, so the AI
    avoids repeating topics. Old entries without a language are treated as English."""
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


def build_prompt(content_type, avoid_text, language):
    hook_rules = """- HOOK (scene 1, first sentence): max 12 words. It must be a bold, surprising CLAIM or STATEMENT that creates instant curiosity or a little shock - e.g. "Your brain deletes 90% of what you see." or "You're using ChatGPT wrong, and it's costing you hours." NEVER open with a question. NEVER start with "Did you know", "Have you ever", "What if", "Imagine", "Today", "Let's talk about" or any greeting/intro.
- Scene 1 must deliver the hook in its first sentence and immediately tease the payoff so viewers stay to the end.
- Keep every scene SHORT: one sentence, 6 to 14 words. Short sentences = fast cuts = higher retention.
- 3 to 5 scenes total, about 40-58 words in total (it will be read aloud at a fast pace in ~16-22 seconds).
- End with a punchy payoff or a one-line takeaway that makes the viewer want to rewatch. Do NOT add a subscribe/follow call-to-action - it is added automatically.
- Each "query" must be a highly SPECIFIC, vivid stock-footage search phrase of 3 to 6 English words that names a concrete subject, an action, and a setting (e.g. "astronaut floating outside space station", "close up hands typing laptop night", "ocean wave crashing rocks slow motion"). NEVER use 1-word or vague queries like "space", "technology", "nature". No brand logos, no named people, no on-screen text.
- Make each scene's query visually DIFFERENT from the others so the video does not feel repetitive."""

    topic_rules_ai = """- Teach ONE genuinely useful, concrete AI tip, trick, or concept per video (e.g. a prompting technique, a way to save time, a common mistake to avoid, or a simple explanation of how AI works), with one short concrete example.
- Query footage should be generic and vivid: people using devices, offices, technology, abstract digital visuals.
- hashtags should mix a couple of broad/high-traffic tags (like "ai", "shorts") with a few specific to this exact tip."""

    topic_rules_random = """- Tell a mini story with build-up and a surprising payoff about ONE true, mind-blowing fact.
- Query footage should be generic and vivid: nature, objects, places, animals.
- hashtags should mix a couple of broad/high-traffic tags (like "shorts", "facts") with a few specific to this exact topic."""

    topic_context = (
        """You write short, punchy scripts for a YouTube Shorts channel
that teaches everyday people practical AI tips, tricks, and beginner concepts
for using AI chatbots and tools (like ChatGPT, Gemini, Claude, or similar) in
daily life, work, or study. Assume the viewer is a curious beginner, not a
programmer."""
        if content_type == "ai_tips" else
        """You write short, punchy scripts for a "mind-blowing facts"
YouTube Shorts channel about surprising true facts (space, science, history,
psychology, nature, animals, or the human body - pick ONE topic at random,
something genuinely surprising and different each time)."""
    )
    topic_rules = topic_rules_ai if content_type == "ai_tips" else topic_rules_random

    if language == "ur":
        return f"""{topic_context}

{avoid_text}

IMPORTANT - this video is for Urdu-speaking viewers, but captions must be
readable by anyone, including Hindi speakers who don't read the Urdu script:
- "title": a short, catchy title written in ROMAN URDU (Urdu typed with English/Latin letters, the way most people type Urdu on WhatsApp/Instagram) - NOT Urdu script, NOT Hindi Devanagari.
- "text" (per scene): the SAME sentence written in proper URDU SCRIPT (Nastaliq/Arabic script). Used only for the voice-over, so it must be correct, natural Urdu script - never shown on screen.
- "caption_roman" (per scene): the SAME sentence transliterated into ROMAN URDU. This is what appears as the on-screen caption.
- "query": in English only, used to search English-language stock footage.
- "hashtags": in English, lowercase.

Return ONLY valid JSON in exactly this shape, no extra commentary:
{{
  "title": "short catchy title in Roman Urdu, under 8 words",
  "hashtags": ["5 to 8 relevant lowercase English hashtags, no # symbol"],
  "scenes": [
    {{"text": "one short sentence in proper Urdu script", "caption_roman": "the same sentence in Roman Urdu", "query": "3-6 word specific English stock-footage search phrase"}}
  ]
}}

Rules:
{hook_rules}
{topic_rules}
- Keep language simple, conversational, and natural for text-to-speech narration.
"""
    else:
        return f"""{topic_context}

{avoid_text}

Write the "title" and every scene's "text" in English.

Return ONLY valid JSON in exactly this shape, no extra commentary:
{{
  "title": "a short catchy title for the video, under 8 words",
  "hashtags": ["5 to 8 relevant lowercase hashtags for this specific video, no # symbol"],
  "scenes": [
    {{"text": "one short spoken sentence", "query": "3-6 word specific stock-footage search phrase"}}
  ]
}}

Rules:
{hook_rules}
{topic_rules}
- Keep language simple, conversational, and natural for text-to-speech narration.
"""


def generate_script_with_ai(content_type, language, max_attempts=3):
    if not GEMINI_KEY:
        print("GEMINI_API_KEY not set, using fallback script.")
        return FALLBACK_SCRIPTS[content_type][language]

    past_titles = get_past_titles(language)
    avoid_text = ""
    if past_titles:
        avoid_text = (
            "Do NOT repeat these topics already covered, pick something different: "
            + "; ".join(past_titles)
        )

    prompt = build_prompt(content_type, avoid_text, language)

    try:
        client = genai.Client(api_key=GEMINI_KEY)
    except Exception as e:
        print(f"Could not create Gemini client, using fallback script: {type(e).__name__}: {e}")
        return FALLBACK_SCRIPTS[content_type][language]

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
                if language == "ur" and not scene.get("caption_roman"):
                    raise ValueError("A scene in the AI response is missing 'caption_roman'.")

            # Enforce a strong hook: reject question / "did you know" openers and retry
            first = data["scenes"][0]["text"].strip()
            if first.endswith(("?", "\u061f")) or first.lower().startswith(
                ("did you know", "have you ever", "what if", "imagine")
            ):
                raise ValueError(f"Weak hook (question opener): {first[:60]}")

            print(f"Gemini script generated successfully on attempt {attempt}/{max_attempts}.")
            return data

        except Exception as e:
            last_error = e
            print(f"[Attempt {attempt}/{max_attempts}] Gemini script generation failed: {type(e).__name__}: {e}")
            if attempt < max_attempts:
                time.sleep(3 * attempt)

    print(f"All {max_attempts} Gemini attempts failed, using fallback script. Last error: {last_error}")
    return FALLBACK_SCRIPTS[content_type][language]


# ----------------------------------------------------------------------------
# Voice (edge-tts, free)
# ----------------------------------------------------------------------------

async def _edge_tts_save(text, voice, rate, out_path):
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(out_path)


def synthesize_speech(text, language, out_path, attempts=3):
    """Generate narration with edge-tts and save it as an .mp3. Retries a few
    times because the free endpoint occasionally hiccups on CI runners."""
    voice = VOICES.get(language, VOICES["en"])
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            asyncio.run(_edge_tts_save(text, voice, NARRATION_RATE, out_path))
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                return out_path
            raise ValueError("edge-tts produced an empty audio file")
        except Exception as e:
            last_error = e
            print(f"[TTS attempt {attempt}/{attempts}] failed: {type(e).__name__}: {e}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"edge-tts failed after {attempts} attempts: {last_error}")


# ----------------------------------------------------------------------------
# Stock footage (Pexels -> Pixabay fallback)
# ----------------------------------------------------------------------------

def search_pexels(query):
    if not PEXELS_KEY:
        return []
    try:
        resp = requests.get(
            "https://api.pexels.com/videos/search",
            headers={"Authorization": PEXELS_KEY},
            params={"query": query, "per_page": 8, "orientation": "portrait"},
            timeout=15,
        )
        if resp.status_code != 200:
            print(f"Pexels returned status {resp.status_code} for '{query}': {resp.text[:150]}")
            return []
        results = []
        for v in resp.json().get("videos", []):
            if v.get("duration", 0) < 3:
                continue
            files = [f for f in v.get("video_files", []) if f.get("file_type") == "video/mp4" and f.get("width")]
            if not files:
                continue
            # Prefer the smallest file that is still >= 720p (fast download on CI)
            good = [f for f in files if min(f["width"], f.get("height") or f["width"]) >= 720]
            pick = (min(good, key=lambda f: f["width"] * (f.get("height") or 0))
                    if good else max(files, key=lambda f: f["width"]))
            results.append((f"pexels:{v['id']}", pick["link"]))
        return results
    except Exception as e:
        print(f"Pexels search failed for '{query}': {type(e).__name__}: {e}")
        return []


def search_pixabay(query):
    if not PIXABAY_KEY:
        return []
    try:
        resp = requests.get(
            "https://pixabay.com/api/videos/",
            params={"key": PIXABAY_KEY, "q": query[:100], "per_page": 10, "safesearch": "true"},
            timeout=15,
        )
        if resp.status_code != 200:
            print(f"Pixabay returned status {resp.status_code} for '{query}': {resp.text[:150]}")
            return []
        results = []
        for hit in resp.json().get("hits", []):
            if hit.get("duration", 0) < 3:
                continue
            vids = hit.get("videos", {})
            for size in ("large", "medium", "small"):
                v = vids.get(size)
                if v and v.get("url"):
                    results.append((f"pixabay:{hit['id']}", v["url"]))
                    break
        return results
    except Exception as e:
        print(f"Pixabay search failed for '{query}': {type(e).__name__}: {e}")
        return []


def find_candidates(query):
    """Pexels first, Pixabay as fallback, then a shortened query on both."""
    queries = [query]
    short = " ".join(query.split()[:2])
    if short and short != query:
        queries.append(short)

    candidates = []
    for q in queries:
        candidates += search_pexels(q)
        if len(candidates) < 2:
            candidates += search_pixabay(q)
        if candidates:
            break
        time.sleep(0.5)
    return [c for c in candidates if c[0] not in USED_CLIP_IDS]


def download_video(url, path):
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    if os.path.getsize(path) < 10_000:
        raise ValueError("Downloaded video file is too small")


def apply_zoom(clip, duration, zoom_in=True, zoom_ratio=0.18):
    """Smooth, eased Ken Burns zoom. Alternates zoom-in / zoom-out per scene
    so consecutive shots never feel the same."""
    def ease(p):
        return p * p * (3 - 2 * p)

    def scale(t):
        p = ease(min(max(t / duration, 0.0), 1.0))
        return 1 + zoom_ratio * (p if zoom_in else (1 - p))

    try:
        zoomed = clip.resized(scale)
        return CompositeVideoClip(
            [zoomed.with_position("center")], size=(WIDTH, HEIGHT)
        ).with_duration(duration)
    except Exception as e:
        print(f"Zoom effect failed, using plain clip: {e}")
        return clip


def fetch_clip(query, duration_needed, index):
    video_file = f"bg_{index}.mp4"
    candidates = find_candidates(query)

    for clip_id, url in candidates[:3]:
        try:
            download_video(url, video_file)
            clip = VideoFileClip(video_file).without_audio()
            w, h = clip.size
            scale = max(HEIGHT / h, WIDTH / w)
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            clip = clip.resized((new_w, new_h))
            x1, y1 = (new_w - WIDTH) // 2, (new_h - HEIGHT) // 2
            clip = clip.cropped(x1=x1, y1=y1, x2=x1 + WIDTH, y2=y1 + HEIGHT)
            clip = clip.with_effects([vfx.MultiplySpeed(SPEED_FACTOR)])

            clips_list, cur_dur = [], 0
            while cur_dur < duration_needed:
                clips_list.append(clip)
                cur_dur += clip.duration
            final_clip = concatenate_videoclips(clips_list).subclipped(0, duration_needed)

            USED_CLIP_IDS.add(clip_id)
            return apply_zoom(final_clip, duration_needed, zoom_in=(index % 2 == 0))
        except Exception as e:
            print(f"Clip {clip_id} failed for '{query}': {type(e).__name__}: {e}")

    print(f"No usable footage for '{query}', using black background.")
    return ImageClip(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)).with_duration(duration_needed)


# ----------------------------------------------------------------------------
# Captions / title
# ----------------------------------------------------------------------------

def _text_clip(**kwargs):
    """TextClip with a small margin so thick strokes aren't clipped; falls
    back gracefully on MoviePy versions without the margin argument."""
    try:
        return TextClip(margin=(30, 30), **kwargs)
    except TypeError:
        return TextClip(**kwargs)


def split_caption_chunks(text, max_words=2, max_chars=14):
    """Group words into 1-2 word chunks (long words stay alone)."""
    chunks, current = [], []
    for word in text.split():
        candidate = current + [word]
        if current and (len(candidate) > max_words or len(" ".join(candidate)) > max_chars):
            chunks.append(" ".join(current))
            current = [word]
        else:
            current = candidate
    if current:
        chunks.append(" ".join(current))
    return chunks


def add_caption(bg_clip, caption_text, duration, is_cta=False):
    """Fast word-by-word captions, uppercase, centered on screen (y ~ 48%).
    Chunk length is proportional to its character count so timing roughly
    follows the speech. Falls back to the plain clip if rendering fails."""
    try:
        chunks = split_caption_chunks(caption_text.upper()) or [caption_text.upper()]
        weights = [len(c) + 2 for c in chunks]
        total = sum(weights)

        caption_clips, start = [], 0.0
        for idx, chunk in enumerate(chunks):
            chunk_dur = duration * weights[idx] / total
            txt = _text_clip(
                font=FONT_PATH,
                text=chunk,
                font_size=88 if is_cta else 84,
                color="yellow" if is_cta else CAPTION_COLORS[idx % len(CAPTION_COLORS)],
                stroke_color="black",
                stroke_width=7,
                method="caption",
                size=(int(WIDTH * 0.88), None),
                text_align="center",
            )
            y = int(HEIGHT * CAPTION_Y_RATIO - txt.h / 2)
            txt = txt.with_duration(chunk_dur).with_start(start).with_position(("center", y))
            caption_clips.append(txt)
            start += chunk_dur

        return CompositeVideoClip([bg_clip] + caption_clips, size=(WIDTH, HEIGHT)).with_duration(duration)
    except Exception as e:
        print(f"Caption failed for text '{caption_text[:30]}...': {e}")
        return bg_clip


def add_title_flash(scene_clip, title_text, flash_duration=1.8):
    """Big bold title card near the top for the first ~1.8s of the first scene
    (kept above center so it never collides with the word-by-word captions)."""
    try:
        flash_len = min(flash_duration, scene_clip.duration)
        title_clip = _text_clip(
            font=FONT_PATH,
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
# Audio extras: music + SFX
# ----------------------------------------------------------------------------

def add_background_music(narration_audio, duration):
    """Mix a quiet, looped royalty-free track under the narration. If the
    'music' folder is missing or empty, narration plays with no music."""
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

        loop_clips, cur_dur = [], 0
        while cur_dur < duration:
            loop_clips.append(music)
            cur_dur += music.duration
        music_full = concatenate_audioclips(loop_clips).subclipped(0, duration)
        music_quiet = music_full.with_volume_scaled(0.12)

        return CompositeAudioClip([narration_audio, music_quiet])
    except Exception as e:
        print(f"Background music failed, continuing without it: {e}")
        return narration_audio


def add_sfx_transitions(audio, transition_times, duration):
    """Overlay a short whoosh at every scene transition if an 'sfx' folder
    with audio files exists (e.g. sfx/whoosh.mp3). Silently skipped otherwise."""
    if not os.path.isdir(SFX_DIR):
        print("No 'sfx' folder found, skipping transition sounds.")
        return audio

    files = [f for f in os.listdir(SFX_DIR) if f.lower().endswith((".mp3", ".wav", ".m4a"))]
    if not files:
        print("'sfx' folder is empty, skipping transition sounds.")
        return audio

    # Prefer a file called whoosh*, otherwise pick any
    preferred = [f for f in files if f.lower().startswith("whoosh")]
    sfx_path = os.path.join(SFX_DIR, random.choice(preferred or files))

    try:
        layers = [audio]
        for t in transition_times:
            if t <= 0 or t >= duration:
                continue
            sfx = AudioFileClip(sfx_path)
            sfx = sfx.subclipped(0, min(1.0, sfx.duration)).with_volume_scaled(0.45)
            start = max(t - 0.12, 0)  # land slightly before the cut
            layers.append(sfx.with_start(start))
        return CompositeAudioClip(layers)
    except Exception as e:
        print(f"SFX overlay failed, continuing without it: {e}")
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


def update_tracker(video_id, title, content_type, language):
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
        "language": language,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "url": f"https://youtube.com/shorts/{video_id}",
    })

    with open(TRACKER_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------

def generate_video():
    content_type, language = get_run_config()

    script = generate_script_with_ai(content_type, language)
    title = script["title"]  # Roman Urdu for ur runs, English for en runs
    cta_scene = CTA_SCENE_UR if language == "ur" else CTA_SCENE_EN
    scenes = list(script["scenes"]) + [cta_scene]

    print(f"Content type: {content_type} | Language: {language} | Topic: {title} | "
          f"Duration window: {MIN_DURATION}-{MAX_DURATION}s | Speed: {SPEED_FACTOR}x")

    audio_clips = []
    video_clips = []
    for i, scene in enumerate(scenes):
        fname = f"part_{i}.mp3"
        synthesize_speech(scene["text"], language, fname)  # "text" is always proper-script for correct pronunciation
        aclip = AudioFileClip(fname)
        audio_clips.append(aclip)

        bg_clip = fetch_clip(scene["query"], aclip.duration, i)
        # on-screen caption: explicit "caption" (CTA) > "caption_roman" (Urdu) > "text" (English)
        caption_text = scene.get("caption") or scene.get("caption_roman") or scene["text"]
        scene_clip = add_caption(bg_clip, caption_text, aclip.duration, is_cta=scene.get("cta", False))
        if i == 0:
            scene_clip = add_title_flash(scene_clip, title)
        video_clips.append(scene_clip)

    # Scene transition timestamps (for SFX)
    transition_times, t = [], 0.0
    for a in audio_clips[:-1]:
        t += a.duration
        transition_times.append(t)

    final_audio = concatenate_audioclips(audio_clips)
    final_video = concatenate_videoclips(video_clips)
    current_duration = min(final_audio.duration, final_video.duration)

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

    final = final_video.with_audio(final_audio)
    output_path = "final_short.mp4"
    final.write_videofile(
        output_path, fps=30, codec="libx264", audio_codec="aac",
        bitrate="5000k", preset="veryfast", threads=4,
    )

    if content_type == "ai_tips":
        base_tags = ["ai", "aitips", "chatgpt", "productivity", "shorts"]
    else:
        base_tags = ["shorts", "facts", "didyouknow"]

    ai_hashtags = [h.strip().lstrip("#").lower() for h in script.get("hashtags", []) if h.strip()]
    all_tags = list(dict.fromkeys(ai_hashtags + base_tags))  # AI's topic-specific tags first, deduped

    hashtag_line = " ".join(f"#{t}" for t in all_tags[:8])
    subscribe_line = (
        "Rozana nayi video ke liye subscribe karein!" if language == "ur"
        else "Subscribe for a new video every single day!"
    )
    description = f"{title}\n\n{subscribe_line}\n\n{hashtag_line}"

    video_id = upload_to_youtube(output_path, title, description, all_tags)
    update_tracker(video_id, title, content_type, language)


if __name__ == "__main__":
    generate_video()
