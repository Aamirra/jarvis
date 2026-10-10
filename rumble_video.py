"""
Long-form (3 minute), landscape 1920x1080 video generator for Rumble.

Pipeline: Gemini script -> edge-tts narration -> unique Pexels/Pixabay footage
-> word-by-word captions -> music + SFX -> H.264/AAC MP4 (Rumble-friendly).

Optional environment variables
  PEXELS_API_KEY, PIXABAY_API_KEY, GEMINI_API_KEY
  UPLOAD_TO_YOUTUBE=1   also upload to YouTube (needs TOKEN_JSON + CLIENT_SECRET_JSON)
  CONTENT_TYPE / VIDEO_LANGUAGE   force "random"|"ai_tips" and "en"|"ur" (for testing)

Rumble has no upload step here: the script writes final_video.mp4 plus
rumble_upload.txt (title / description / tags) ready to paste into Rumble.
"""
import os
import json
import math
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
    CompositeAudioClip, vfx, afx
)

from google import genai
from google.genai import types

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------

WIDTH, HEIGHT = 1920, 1080          # landscape 16:9, Rumble's native size
FPS = 30
MIN_DURATION = 170.0
MAX_DURATION = 190.0                # padded to MIN / trimmed to MAX only if the natural length falls outside

PEXELS_KEY = os.environ.get("PEXELS_API_KEY")
PIXABAY_KEY = os.environ.get("PIXABAY_API_KEY")  # optional, free: https://pixabay.com/api/docs/
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.1-flash-lite"

TRACKER_FILE = "upload_tracker.json"
MUSIC_DIR = "music"  # royalty-free .mp3 files; one is picked at random each run
SFX_DIR = "sfx"      # short transition sounds (e.g. whoosh.mp3); optional
OUTPUT_PATH = "final_video.mp4"
METADATA_PATH = "rumble_upload.txt"

# Script shape
MIN_SCENES, MAX_SCENES = 12, 15
TARGET_WORDS = (470, 530)   # what we ask Gemini for
ACCEPT_WORDS = (430, 600)   # what we accept before asking Gemini to retry

# Pacing: narration is sped up via edge-tts' native rate (keeps pitch natural),
# and stock footage is sped up with MoviePy's MultiplySpeed so visuals feel snappy too.
SPEED_FACTOR = 1.12
MAX_NARRATION_SPEED = 1.35  # upper limit when we have to speed narration up to fit MAX_DURATION


def rate_for_speed(speed):
    return f"+{int(math.ceil((speed - 1) * 100))}%"


VOICES = {
    "en": "en-US-ChristopherNeural",
    "ur": "ur-PK-AsadNeural",
}

# Footage: every scene is cut into shots of at most SEGMENT_MAX seconds, and every
# shot is a DIFFERENT stock clip, which keeps a 3-minute video visually fresh.
SEGMENT_MAX = 7.0
MIN_CLIP_SECONDS = 2.0
MAX_SCENE_GAP = 1.2         # max silent "breath" added after a scene when padding to MIN_DURATION

# Captions: lower third is the natural place on a landscape screen.
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
CAPTION_Y_RATIO = 0.80
CAPTION_FONT_SIZE = 76
CAPTION_STROKE = 6
CAPTION_WIDTH_RATIO = 0.80
CAPTION_COLORS = ["white", "yellow"]  # alternates per word chunk for a pop effect
TITLE_FONT_SIZE = 62
TITLE_Y_RATIO = 0.12

# 6 runs per day, one every 4 hours (UTC). Workflow cron: '0 */4 * * *'.
# 3-minute videos are heavier than Shorts: lower the cron frequency if CI minutes matter.
RUN_SCHEDULE = {
    0: ("random", "en"),
    4: ("ai_tips", "ur"),
    8: ("random", "ur"),
    12: ("ai_tips", "en"),
    16: ("random", "en"),
    20: ("ai_tips", "ur"),
}

USED_CLIP_IDS = set()  # a stock clip id may NEVER appear twice in one video
TEMP_FILES = []


def get_run_config():
    """Decide today's content type + language from the UTC hour. Snaps to the
    nearest scheduled hour so a slightly delayed workflow run still picks a
    sensible slot instead of crashing. Env vars can override for testing."""
    forced_type = os.environ.get("CONTENT_TYPE")
    forced_lang = os.environ.get("VIDEO_LANGUAGE")
    hour = datetime.now(timezone.utc).hour
    closest_hour = min(RUN_SCHEDULE.keys(), key=lambda h: min(abs(hour - h), 24 - abs(hour - h)))
    content_type, language = RUN_SCHEDULE[closest_hour]
    if forced_type in ("random", "ai_tips"):
        content_type = forced_type
    if forced_lang in ("en", "ur"):
        language = forced_lang
    return content_type, language


# Urdu runs: "text" stays in proper Urdu script because TTS needs real Urdu
# script to pronounce it correctly - that field is only ever used for the
# voice-over, never shown on screen. "caption" is Roman Urdu.
CTA_SCENE_EN = {
    "text": "Follow for a brand new one every single day.",
    "query": "neon gradient abstract motion background",
    "backup_queries": ["colorful light trails dark background", "abstract particles glowing blue loop"],
    "cta": True,
    "caption": "FOLLOW FOR A NEW ONE EVERY DAY!",
}

CTA_SCENE_UR = {
    "text": "روزانہ نئی ویڈیو کے لیے فالو کریں۔",
    "query": "neon gradient abstract motion background",
    "backup_queries": ["colorful light trails dark background", "abstract particles glowing blue loop"],
    "cta": True,
    "caption": "ROZANA NAYI VIDEO KE LIYE FOLLOW KAREIN!",
}

# Used only if AI script generation fails completely, so the pipeline never crashes.
# NOTE: these are short, so a video built from them will be shorter than MIN_DURATION.
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

def get_past_titles(language, limit=40):
    """Titles of previously generated videos in the same language, so the AI
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
    structure_rules = f"""- Write between {MIN_SCENES} and {MAX_SCENES} scenes (aim for 13). This is a 3-minute video, so it needs depth: a real story, not a list of one-liners.
- Each scene is 2 to 3 short spoken sentences, 28 to 42 words. The TOTAL spoken words across all scenes must be between {TARGET_WORDS[0]} and {TARGET_WORDS[1]} (it is read aloud at a fast pace in about 3 minutes).
- Structure: scene 1 = hook; scene 2 = context that makes the viewer care; middle scenes = build-up where EACH scene adds a NEW surprising detail, example, or step (no repeating earlier points); second-to-last scene = the biggest twist or payoff; last scene = a punchy takeaway worth rewatching. Do NOT add a subscribe/follow call-to-action - it is added automatically.
- Keep viewers watching: every 3 or 4 scenes, tease what is coming next ("But the strangest part is still ahead.").
- HOOK (scene 1, first sentence): max 12 words. It must be a bold, surprising CLAIM or STATEMENT that creates instant curiosity or a little shock - e.g. "Your brain deletes 90% of what you see." or "You're using ChatGPT wrong, and it's costing you hours." NEVER open with a question. NEVER start with "Did you know", "Have you ever", "What if", "Imagine", "Today", "Let's talk about" or any greeting/intro.
- STOCK FOOTAGE QUERIES (very important): every scene needs a "query" that is a highly SPECIFIC, vivid stock-footage search phrase of 3 to 6 English words naming a concrete subject, an action and a setting (e.g. "astronaut floating outside space station", "close up hands typing laptop night", "ocean wave crashing rocks slow motion"). Choose things that really exist as stock video (people, places, animals, objects, nature, cities), not abstract concepts.
- EVERY query must be COMPLETELY UNIQUE across the whole video: different main subject, different setting, different camera angle. Never repeat a query, never reword an earlier one, never reuse the same main noun in two queries. NEVER use 1-2 word or vague queries like "space", "technology", "nature". No brand logos, no named people, no on-screen text.
- Each scene also needs "backup_queries": 2 more specific English queries (3 to 6 words) that fit the same scene but show a different subject, also unique across the whole video. They are used if the main query finds no footage."""

    topic_rules_ai = """- Teach ONE genuinely useful, concrete AI skill per video (e.g. a prompting technique, a time-saving workflow, a common mistake to avoid, or a clear explanation of how AI works), with at least two short concrete examples.
- Footage should be generic and vivid: people using devices, offices, workspaces, cities, technology, abstract digital visuals.
- hashtags should mix a couple of broad/high-traffic tags (like "ai", "chatgpt") with a few specific to this exact tip."""

    topic_rules_random = """- Tell a mini documentary-style story with build-up and a surprising payoff about ONE true, mind-blowing topic. Use accurate, checkable facts only.
- Footage should be generic and vivid: nature, objects, places, animals, science, cities.
- hashtags should mix a couple of broad/high-traffic tags (like "facts", "science") with a few specific to this exact topic."""

    topic_context = (
        """You write engaging, well-paced scripts for a 3-minute YouTube/Rumble video channel
that teaches everyday people practical AI tips, tricks, and beginner concepts
for using AI chatbots and tools (like ChatGPT, Gemini, Claude, or similar) in
daily life, work, or study. Assume the viewer is a curious beginner, not a
programmer."""
        if content_type == "ai_tips" else
        """You write engaging, well-paced scripts for a 3-minute "mind-blowing facts"
video channel about surprising true facts (space, science, history,
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
- "text" (per scene): the SAME content written in proper URDU SCRIPT (Nastaliq/Arabic script). Used only for the voice-over, so it must be correct, natural Urdu script - never shown on screen.
- "caption_roman" (per scene): the SAME content transliterated into ROMAN URDU. This is what appears as the on-screen caption.
- "query" and "backup_queries": in English only, used to search English-language stock footage.
- "hashtags": in English, lowercase.

Return ONLY valid JSON in exactly this shape, no extra commentary:
{{
  "title": "short catchy title in Roman Urdu, under 8 words",
  "hashtags": ["5 to 8 relevant lowercase English hashtags, no # symbol"],
  "scenes": [
    {{"text": "2-3 short sentences in proper Urdu script", "caption_roman": "the same sentences in Roman Urdu", "query": "3-6 word specific unique English stock-footage search phrase", "backup_queries": ["another unique 3-6 word phrase", "a third unique 3-6 word phrase"]}}
  ]
}}

Rules:
{structure_rules}
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
    {{"text": "2-3 short spoken sentences", "query": "3-6 word specific unique stock-footage search phrase", "backup_queries": ["another unique 3-6 word phrase", "a third unique 3-6 word phrase"]}}
  ]
}}

Rules:
{structure_rules}
{topic_rules}
- Keep language simple, conversational, and natural for text-to-speech narration.
"""


def _query_words(q):
    return set(w for w in "".join(c.lower() if c.isalnum() or c == " " else " " for c in q).split() if len(w) > 2)


def check_script_quality(data):
    """Soft quality checks. Returns a list of human-readable problems (empty = good).
    The problems are fed back to Gemini on retry."""
    problems = []
    scenes = data["scenes"]
    n = len(scenes)
    if not (MIN_SCENES <= n <= MAX_SCENES):
        problems.append(f"it had {n} scenes but it must have between {MIN_SCENES} and {MAX_SCENES}")

    words = sum(len(s["text"].split()) for s in scenes)
    if not (ACCEPT_WORDS[0] <= words <= ACCEPT_WORDS[1]):
        problems.append(f"the total narration was {words} words but it must be between {TARGET_WORDS[0]} and {TARGET_WORDS[1]}")

    first = scenes[0]["text"].strip()
    if first.endswith(("?", "\u061f")) or first.lower().startswith(
        ("did you know", "have you ever", "what if", "imagine")
    ):
        problems.append("the opening was a question/'did you know' style hook; it must be a bold statement")

    sets = [_query_words(s["query"]) for s in scenes]
    for i, s in enumerate(scenes):
        if len(s["query"].split()) < 3:
            problems.append(f"scene {i + 1} query '{s['query']}' is too vague (needs 3-6 specific words)")
    for i in range(n):
        for j in range(i + 1, n):
            a, b = sets[i], sets[j]
            if a and b and len(a & b) / len(a | b) > 0.5:
                problems.append(f"queries of scene {i + 1} and scene {j + 1} are too similar; every query must be completely unique")
    return problems[:8]


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

    base_prompt = build_prompt(content_type, avoid_text, language)

    try:
        client = genai.Client(api_key=GEMINI_KEY)
    except Exception as e:
        print(f"Could not create Gemini client, using fallback script: {type(e).__name__}: {e}")
        return FALLBACK_SCRIPTS[content_type][language]

    last_error = None
    best = None            # (problem_count, data) of the best usable-but-imperfect script
    feedback = ""
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=base_prompt + feedback,
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.9),
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
                bq = scene.get("backup_queries")
                scene["backup_queries"] = [q for q in bq if isinstance(q, str) and q.strip()] if isinstance(bq, list) else []

            problems = check_script_quality(data)
            if not problems:
                print(f"Gemini script generated successfully on attempt {attempt}/{max_attempts} "
                      f"({len(data['scenes'])} scenes).")
                return data

            print(f"[Attempt {attempt}/{max_attempts}] Script rejected: {'; '.join(problems)}")
            if len(data["scenes"]) >= 8 and (best is None or len(problems) < best[0]):
                best = (len(problems), data)
            feedback = ("\n\nYour previous attempt was rejected because: " + "; ".join(problems)
                        + ". Write a new script and fix every one of these problems.")

        except Exception as e:
            last_error = e
            print(f"[Attempt {attempt}/{max_attempts}] Gemini script generation failed: {type(e).__name__}: {e}")
        if attempt < max_attempts:
            time.sleep(3 * attempt)

    if best is not None:
        print("Using the best imperfect Gemini script (length is still handled by the padding/trimming logic).")
        return best[1]

    print(f"All {max_attempts} Gemini attempts failed, using fallback script. Last error: {last_error}")
    return FALLBACK_SCRIPTS[content_type][language]


# ----------------------------------------------------------------------------
# Voice (edge-tts, free)
# ----------------------------------------------------------------------------

async def _edge_tts_save(text, voice, rate, out_path):
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(out_path)


def synthesize_speech(text, language, out_path, rate, attempts=3):
    """Generate narration with edge-tts and save it as an .mp3. Retries a few
    times because the free endpoint occasionally hiccups on CI runners."""
    voice = VOICES.get(language, VOICES["en"])
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            asyncio.run(_edge_tts_save(text, voice, rate, out_path))
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                return out_path
            raise ValueError("edge-tts produced an empty audio file")
        except Exception as e:
            last_error = e
            print(f"[TTS attempt {attempt}/{attempts}] failed: {type(e).__name__}: {e}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"edge-tts failed after {attempts} attempts: {last_error}")


def synthesize_all(scenes, language, rate):
    clips = []
    for i, scene in enumerate(scenes):
        fname = f"part_{i}.mp3"
        if fname not in TEMP_FILES:
            TEMP_FILES.append(fname)
        # "text" is always proper-script so pronunciation is correct
        synthesize_speech(scene["text"], language, fname, rate)
        clips.append(AudioFileClip(fname))
    return clips


# ----------------------------------------------------------------------------
# Stock footage (Pexels -> Pixabay fallback), unique clips only
# ----------------------------------------------------------------------------

def _pick_pexels_file(files):
    """Prefer a Full HD file (smallest one >= 1920 wide) so downloads stay fast on CI."""
    def h(f):
        return f.get("height") or 0
    wide = [f for f in files if f["width"] >= 1280]
    if not wide:
        return max(files, key=lambda f: f["width"])
    ideal = [f for f in wide if 1920 <= f["width"] <= 2560]
    pool = ideal or wide
    return min(pool, key=lambda f: f["width"] * h(f))


def search_pexels(query, page=1):
    if not PEXELS_KEY:
        return []
    try:
        resp = requests.get(
            "https://api.pexels.com/videos/search",
            headers={"Authorization": PEXELS_KEY},
            params={"query": query, "per_page": 15, "page": page, "orientation": "landscape"},
            timeout=15,
        )
        if resp.status_code != 200:
            print(f"Pexels returned status {resp.status_code} for '{query}': {resp.text[:150]}")
            return []
        results = []
        for v in resp.json().get("videos", []):
            if v.get("duration", 0) < 4:
                continue
            files = [f for f in v.get("video_files", []) if f.get("file_type") == "video/mp4" and f.get("width")]
            if not files:
                continue
            results.append((f"pexels:{v['id']}", _pick_pexels_file(files)["link"]))
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
            params={"key": PIXABAY_KEY, "q": query[:100], "per_page": 20, "safesearch": "true"},
            timeout=15,
        )
        if resp.status_code != 200:
            print(f"Pixabay returned status {resp.status_code} for '{query}': {resp.text[:150]}")
            return []
        results = []
        for hit in resp.json().get("hits", []):
            if hit.get("duration", 0) < 4:
                continue
            vids = hit.get("videos", {})
            for size in ("large", "medium", "small"):  # "large" is 1920x1080 on Pixabay
                v = vids.get(size)
                if v and v.get("url"):
                    results.append((f"pixabay:{hit['id']}", v["url"]))
                    break
        return results
    except Exception as e:
        print(f"Pixabay search failed for '{query}': {type(e).__name__}: {e}")
        return []


def gather_candidates(query):
    """Unused candidates for one query: Pexels p1, Pixabay, Pexels p2, then the
    same on a shortened query. Clips already used in this video are dropped
    immediately, so a clip id can never come back."""
    variants = [query]
    short = " ".join(query.split()[:3])
    if short and short != query:
        variants.append(short)

    found, seen = [], set()

    def add(batch):
        for cid, url in batch:
            if cid in USED_CLIP_IDS or cid in seen:
                continue
            seen.add(cid)
            found.append((cid, url))

    for q in variants:
        add(search_pexels(q, 1))
        if len(found) < 4:
            add(search_pixabay(q))
        if len(found) < 4:
            add(search_pexels(q, 2))
        if len(found) >= 4:
            break
        time.sleep(0.4)
    return found


def download_video(url, path):
    with requests.get(url, stream=True, timeout=45) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    if os.path.getsize(path) < 10_000:
        raise ValueError("Downloaded video file is too small")


def apply_zoom(clip, duration, zoom_in=True, zoom_ratio=0.08):
    """Smooth, eased Ken Burns zoom. Alternates zoom-in / zoom-out per shot
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


def gradient_clip(duration, variant=0):
    """Colourful gradient used only when no footage at all can be found (never a black screen)."""
    x = np.linspace(0, 1, WIDTH, dtype=np.float32)[None, :]
    y = np.linspace(0, 1, HEIGHT, dtype=np.float32)[:, None]
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    if variant % 2 == 0:
        frame[..., 0] = 25 + 60 * x
        frame[..., 1] = 15 + 30 * y
        frame[..., 2] = 70 + 110 * (1 - x) * y
    else:
        frame[..., 0] = 70 + 110 * (1 - x) * y
        frame[..., 1] = 20 + 50 * x
        frame[..., 2] = 40 + 80 * y
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    return ImageClip(frame).with_duration(duration)


def load_and_fit_clip(url, target, tag):
    """Download, scale-to-cover 1920x1080, speed up, and cut to at most `target` seconds."""
    path = f"bg_{tag}.mp4"
    TEMP_FILES.append(path)
    download_video(url, path)
    clip = VideoFileClip(path).without_audio()
    w, h = clip.size
    scale = max(HEIGHT / h, WIDTH / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    clip = clip.resized((new_w, new_h))
    x1, y1 = (new_w - WIDTH) // 2, (new_h - HEIGHT) // 2
    clip = clip.cropped(x1=x1, y1=y1, x2=x1 + WIDTH, y2=y1 + HEIGHT)
    clip = clip.with_effects([vfx.MultiplySpeed(SPEED_FACTOR)])
    if clip.duration < MIN_CLIP_SECONDS:
        raise ValueError(f"clip too short after speed-up ({clip.duration:.1f}s)")
    length = min(target, clip.duration - 0.05)
    return clip.subclipped(0, length)


def fetch_footage(query, duration, scene_index, backup_queries=None):
    """Build `duration` seconds of footage for one scene from several DIFFERENT
    stock clips (each shot <= SEGMENT_MAX). A clip id is claimed the moment it is
    considered, so it can never be reused anywhere in the video. If every query is
    exhausted, the rest is filled with a frozen last frame or a gradient (no black)."""
    pending_queries = [q for q in [query] + list(backup_queries or []) if q]
    candidates = []
    pieces, remaining, shot = [], duration, 0

    while remaining > 0.05:
        if not candidates:
            if not pending_queries:
                break
            candidates = gather_candidates(pending_queries.pop(0))
            continue

        clip_id, url = candidates.pop(0)
        if clip_id in USED_CLIP_IDS:
            continue
        USED_CLIP_IDS.add(clip_id)

        target = min(remaining, SEGMENT_MAX)
        if remaining - target < 1.5:      # avoid a tiny leftover shot
            target = remaining
        try:
            raw = load_and_fit_clip(url, target, f"{scene_index}_{shot}")
            piece = apply_zoom(raw, raw.duration, zoom_in=((scene_index + shot) % 2 == 0))
            pieces.append(piece)
            remaining -= piece.duration
            shot += 1
        except Exception as e:
            print(f"Clip {clip_id} failed for scene {scene_index}: {type(e).__name__}: {e}")

    if remaining > 0.001:
        if pieces and remaining <= 3.0:
            last = pieces[-1]
            frame = last.get_frame(max(last.duration - 0.05, 0))
            pieces.append(ImageClip(frame).with_duration(remaining))
            print(f"Scene {scene_index}: froze last frame for {remaining:.1f}s (footage ran out).")
        else:
            print(f"Scene {scene_index}: no more unique footage, using gradient for {remaining:.1f}s.")
            pieces.append(apply_zoom(gradient_clip(remaining, scene_index), remaining, zoom_in=True))

    return pieces[0] if len(pieces) == 1 else concatenate_videoclips(pieces)


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


def split_caption_chunks(text, max_words=3, max_chars=22):
    """Group words into short chunks (long words stay alone)."""
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


def add_caption(bg_clip, caption_text, speech_duration, total_duration, is_cta=False):
    """Fast word-group captions, uppercase, in the lower third of a landscape screen.
    Chunk length is proportional to its character count so timing roughly follows the
    speech; the last chunk stays up through any silent pause. Falls back to the plain
    clip if rendering fails."""
    try:
        chunks = split_caption_chunks(caption_text.upper()) or [caption_text.upper()]
        weights = [len(c) + 2 for c in chunks]
        total = sum(weights)

        caption_clips, start = [], 0.0
        for idx, chunk in enumerate(chunks):
            chunk_dur = speech_duration * weights[idx] / total
            if idx == len(chunks) - 1:
                chunk_dur = max(total_duration - start, 0.1)
            txt = _text_clip(
                font=FONT_PATH,
                text=chunk,
                font_size=CAPTION_FONT_SIZE + (6 if is_cta else 0),
                color="yellow" if is_cta else CAPTION_COLORS[idx % len(CAPTION_COLORS)],
                stroke_color="black",
                stroke_width=CAPTION_STROKE,
                method="caption",
                size=(int(WIDTH * CAPTION_WIDTH_RATIO), None),
                text_align="center",
            )
            y = int(HEIGHT * CAPTION_Y_RATIO - txt.h / 2)
            txt = txt.with_duration(chunk_dur).with_start(start).with_position(("center", y))
            caption_clips.append(txt)
            start += chunk_dur

        return CompositeVideoClip([bg_clip] + caption_clips, size=(WIDTH, HEIGHT)).with_duration(total_duration)
    except Exception as e:
        print(f"Caption failed for text '{caption_text[:30]}...': {e}")
        return bg_clip


def add_title_flash(scene_clip, title_text, flash_duration=4.0):
    """Bold title near the top for the first few seconds of the first scene
    (kept above the lower-third captions so they never collide)."""
    try:
        flash_len = min(flash_duration, scene_clip.duration)
        title_clip = _text_clip(
            font=FONT_PATH,
            text=title_text.upper(),
            font_size=TITLE_FONT_SIZE,
            color="white",
            stroke_color="black",
            stroke_width=6,
            method="caption",
            size=(int(WIDTH * 0.80), None),
            text_align="center",
        )
        y = int(HEIGHT * TITLE_Y_RATIO - title_clip.h / 2)
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
        music_quiet = music_full.with_volume_scaled(0.10)
        try:
            music_quiet = music_quiet.with_effects([afx.AudioFadeOut(2.0)])
        except Exception:
            pass

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

    preferred = [f for f in files if f.lower().startswith("whoosh")]
    sfx_path = os.path.join(SFX_DIR, random.choice(preferred or files))

    try:
        layers = [audio]
        for t in transition_times:
            if t <= 0 or t >= duration:
                continue
            sfx = AudioFileClip(sfx_path)
            sfx = sfx.subclipped(0, min(1.0, sfx.duration)).with_volume_scaled(0.40)
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
# Timing helpers
# ----------------------------------------------------------------------------

def plan_gaps(durations):
    """Silent pauses added after each scene so the total reaches MIN_DURATION without
    freezing the picture. Audio and video of a scene always get the same length, so
    sync can never drift. If the script is far too short (e.g. the emergency fallback
    script) we don't stretch it artificially."""
    n = len(durations)
    total = sum(durations)
    gaps = [0.0] * n
    if total >= MIN_DURATION:
        return gaps
    if total < MIN_DURATION * 0.6:
        print(f"WARNING: narration is only {total:.0f}s (script too short, probably the fallback). "
              f"Not padding to {MIN_DURATION:.0f}s.")
        return gaps
    pad = MIN_DURATION - total + 0.5
    per = min(pad / n, MAX_SCENE_GAP)
    gaps = [per] * n
    leftover = pad - per * n
    if leftover > 0:
        gaps[-1] += leftover          # extra goes to the end card
    print(f"Padding {pad:.1f}s of natural pauses across {n} scenes to reach {MIN_DURATION:.0f}s.")
    return gaps


def match_av(video, audio):
    """Make video and audio the same length: extend video with a last-frame hold
    or audio with silence (never black, never a crash)."""
    diff = audio.duration - video.duration
    if diff > 0.02:
        last = video.get_frame(max(video.duration - 0.05, 0))
        video = concatenate_videoclips([video, ImageClip(last).with_duration(diff)])
    elif diff < -0.02:
        audio = concatenate_audioclips([audio, silence_clip(-diff)])
    return video, audio


# ----------------------------------------------------------------------------
# Publishing
# ----------------------------------------------------------------------------

def save_rumble_metadata(title, description, tags):
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        f.write(f"TITLE:\n{title}\n\nDESCRIPTION:\n{description}\n\nTAGS:\n{', '.join(tags)}\n")
    print(f"Saved Rumble upload details to {METADATA_PATH}")


def get_youtube_client():
    """Build an authenticated YouTube API client from the TOKEN_JSON and
    CLIENT_SECRET_JSON secrets, refreshing the access token if needed."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

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
    from googleapiclient.http import MediaFileUpload
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
    print(f"Uploaded successfully: https://youtube.com/watch?v={video_id}")
    return video_id


def update_tracker(video_id, title, content_type, language, url=None):
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
        "url": url,
    })

    with open(TRACKER_FILE, "w") as f:
        json.dump(data, f, indent=2)


def cleanup_temp_files():
    for path in TEMP_FILES:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------

def generate_video():
    USED_CLIP_IDS.clear()
    TEMP_FILES.clear()

    content_type, language = get_run_config()

    script = generate_script_with_ai(content_type, language)
    title = script["title"]  # Roman Urdu for ur runs, English for en runs
    cta_scene = CTA_SCENE_UR if language == "ur" else CTA_SCENE_EN
    scenes = list(script["scenes"]) + [cta_scene]

    print(f"Content type: {content_type} | Language: {language} | Topic: {title} | "
          f"Scenes: {len(scenes)} | Target: {MIN_DURATION:.0f}-{MAX_DURATION:.0f}s | {WIDTH}x{HEIGHT}")

    # 1) Narration first, so we know the real length before building any video.
    speed = SPEED_FACTOR
    audio_clips = synthesize_all(scenes, language, rate_for_speed(speed))
    total = sum(a.duration for a in audio_clips)
    print(f"Narration length: {total:.1f}s")

    # 2) Too long? Re-record once slightly faster instead of chopping off the ending.
    limit = MAX_DURATION - 1.0
    if total > limit:
        new_speed = min(speed * total / limit, MAX_NARRATION_SPEED)
        if new_speed > speed + 0.01:
            print(f"Narration too long, re-recording at {new_speed:.2f}x ({rate_for_speed(new_speed)}).")
            for a in audio_clips:
                a.close()
            audio_clips = synthesize_all(scenes, language, rate_for_speed(new_speed))
            total = sum(a.duration for a in audio_clips)
            print(f"New narration length: {total:.1f}s")

    # 3) Too short? Add natural pauses (same length added to audio and video of each scene).
    gaps = plan_gaps([a.duration for a in audio_clips])

    scene_audios, video_clips, transition_times = [], [], []
    elapsed = 0.0
    for i, scene in enumerate(scenes):
        a = audio_clips[i]
        scene_len = a.duration + gaps[i]

        bg_clip = fetch_footage(scene["query"], scene_len, i, scene.get("backup_queries"))
        # on-screen caption: explicit "caption" (CTA) > "caption_roman" (Urdu) > "text" (English)
        caption_text = scene.get("caption") or scene.get("caption_roman") or scene["text"]
        scene_clip = add_caption(bg_clip, caption_text, a.duration, scene_len, is_cta=scene.get("cta", False))
        if i == 0:
            scene_clip = add_title_flash(scene_clip, title)
        video_clips.append(scene_clip)

        scene_audios.append(concatenate_audioclips([a, silence_clip(gaps[i])]) if gaps[i] > 0.01 else a)

        elapsed += scene_len
        if i < len(scenes) - 1:
            transition_times.append(elapsed)

    final_audio = concatenate_audioclips(scene_audios)
    final_video = concatenate_videoclips(video_clips)
    final_video, final_audio = match_av(final_video, final_audio)

    final_duration = min(final_audio.duration, final_video.duration)
    if final_duration > MAX_DURATION:
        print(f"Still {final_duration:.1f}s long, trimming to {MAX_DURATION:.0f}s with a short audio fade.")
        final_duration = MAX_DURATION
        final_audio = final_audio.subclipped(0, final_duration)
        try:
            final_audio = final_audio.with_effects([afx.AudioFadeOut(0.8)])
        except Exception:
            pass
    else:
        final_audio = final_audio.subclipped(0, final_duration)
    final_video = final_video.subclipped(0, final_duration)

    final_audio = add_background_music(final_audio, final_duration)
    final_audio = add_sfx_transitions(final_audio, transition_times, final_duration)
    final_audio = final_audio.with_duration(final_duration)

    final = final_video.with_audio(final_audio)
    final.write_videofile(
        OUTPUT_PATH, fps=FPS, codec="libx264", audio_codec="aac",
        bitrate="8000k", audio_bitrate="192k", audio_fps=48000,
        preset="veryfast", threads=4,
        ffmpeg_params=["-profile:v", "high", "-level", "4.1", "-movflags", "+faststart"],
    )
    print(f"Rendered {OUTPUT_PATH}: {final_duration:.1f}s, {WIDTH}x{HEIGHT}, "
          f"{len(USED_CLIP_IDS)} unique stock clips used.")

    # Metadata
    if content_type == "ai_tips":
        base_tags = ["ai", "aitips", "chatgpt", "productivity", "tech"]
    else:
        base_tags = ["facts", "didyouknow", "science", "mindblowing"]

    ai_hashtags = [h.strip().lstrip("#").lower() for h in script.get("hashtags", []) if h.strip()]
    all_tags = list(dict.fromkeys(ai_hashtags + base_tags))  # AI's topic-specific tags first, deduped

    hashtag_line = " ".join(f"#{t}" for t in all_tags[:8])
    follow_line = (
        "Rozana nayi video ke liye follow karein!" if language == "ur"
        else "Follow for a new video every single day!"
    )
    description = f"{title}\n\n{follow_line}\n\n{hashtag_line}"

    save_rumble_metadata(title, description, all_tags)

    video_id, url = f"local-{int(time.time())}", None
    if os.environ.get("UPLOAD_TO_YOUTUBE") == "1":
        try:
            video_id = upload_to_youtube(OUTPUT_PATH, title, description, all_tags)
            url = f"https://youtube.com/watch?v={video_id}"
        except Exception as e:
            print(f"YouTube upload failed (video is still saved locally): {type(e).__name__}: {e}")
    update_tracker(video_id, title, content_type, language, url)

    cleanup_temp_files()


if __name__ == "__main__":
    generate_video()
