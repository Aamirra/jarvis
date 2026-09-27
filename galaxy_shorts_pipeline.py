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
from gtts import gTTS  # kept only as an emergency fallback if edge-tts fails
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
MIN_DURATION = 30.0
MAX_DURATION = 40.0  # each video's final length is picked randomly between these
PEXELS_KEY = os.environ.get("PEXELS_API_KEY")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.1-flash-lite"  # cheap + fast, plenty for short scripts, free-tier friendly

# Microsoft Edge's free neural TTS voice (no API key, no account, no cost).
# Browse more options with: edge-tts --list-voices
# A few good English narrator picks: en-US-AndrewNeural (male), en-US-GuyNeural
# (male), en-US-AriaNeural (female), en-US-EmmaNeural (female).
EDGE_TTS_VOICE = os.environ.get("EDGE_TTS_VOICE", "en-US-AndrewNeural")

TRACKER_FILE = "upload_tracker.json"
MUSIC_DIR = "music"  # put a few royalty-free .mp3 files here; one is picked at random each run

# YouTube category per content type (Education fits AI tips; Science & Tech
# fits "did you know" facts better than the old one-size-fits-all Education).
CATEGORY_IDS = {"ai_tips": "27", "random": "28"}

# Font used for on-screen captions (installed via apt in the workflow: fonts-dejavu-core).
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# One run every 4 hours (UTC), alternating between the two content types.
# Update your workflow's cron to '0 */4 * * *' so it actually triggers here.
# Want more videos per day? Add more hours below (e.g. every 2 hours for 12/day)
# and update the cron schedule to match.
RUN_SCHEDULE = {
    0: "random",
    4: "ai_tips",
    8: "random",
    12: "ai_tips",
    16: "random",
    20: "ai_tips",
}


def get_run_config():
    """Decide today's content type from the UTC hour. Snaps to the nearest
    scheduled hour so a slightly delayed workflow run (common with GitHub
    Actions cron) still picks a sensible slot instead of crashing."""
    hour = datetime.now(timezone.utc).hour
    closest_hour = min(RUN_SCHEDULE.keys(), key=lambda h: min(abs(hour - h), 24 - abs(hour - h)))
    return RUN_SCHEDULE[closest_hour]


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
        ],
    },
    "ai_tips": {
        "title": "This One Prompt Trick Changes Everything",
        "hashtags": ["ai", "aitips", "chatgpt", "prompting", "productivity"],
        "scenes": [
            {"text": "Most people use AI chatbots completely wrong, and it's costing them much better answers.", "query": "person typing laptop screen"},
            {"text": "Here's a simple trick: instead of just asking a question, show the AI an example of what you want first.", "query": "hands typing keyboard closeup"},
            {"text": "This is called few shot prompting, and it works because AI learns better from examples than from instructions alone.", "query": "digital technology abstract lights"},
            {"text": "Try this in your very next chat with any AI tool, and watch how much better the answer gets.", "query": "smartphone chat app screen"},
        ],
    },
}


def get_past_titles(limit=20):
    """Read titles of previously uploaded videos, so we can ask the AI to
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
    topic_rules_ai = """- 4 to 6 scenes total.
- Teach ONE genuinely useful, concrete AI tip, trick, or concept per video (e.g. a prompting technique, a way to save time, a common mistake to avoid, or a simple explanation of how AI works).
- All scenes combined should read aloud in about 22-30 seconds (roughly 60-85 words total).
- The very first sentence must be a bold, scroll-stopping hook - a surprising claim, mistake, or question - written to stop someone mid-scroll in the first 2 seconds. Then explain the tip clearly, then give one short concrete example.
- Each "query" must describe generic, vivid, specific stock video footage (people using devices, offices, technology, abstract digital visuals) - never named apps' logos or real people - so it closely matches the sentence and can be found on a stock footage site.
- hashtags should mix a couple of broad/high-traffic tags (like "ai", "shorts") with a few specific to this exact tip, to help discovery."""

    topic_rules_random = """- 4 to 6 scenes total.
- All scenes combined should read aloud in about 22-30 seconds (roughly 60-85 words total).
- The very first sentence must be a bold, scroll-stopping hook - a surprising claim or question - written to stop someone mid-scroll in the first 2 seconds. Then the rest should flow like a mini story with build-up and a surprising payoff.
- Each "query" must describe generic, vivid, specific stock video footage (nature, objects, places, animals) - never named people or brands - so it closely matches the sentence and can be found on a stock footage site.
- hashtags should mix a couple of broad/high-traffic tags (like "shorts", "facts") with a few specific to this exact topic, to help discovery."""

    topic_context = (
        """You write short, punchy scripts for a YouTube Shorts channel
that teaches everyday people practical AI tips, tricks, and beginner concepts
for using AI chatbots and tools (like ChatGPT, Gemini, Claude, or similar) in
daily life, work, or study. Assume the viewer is a curious beginner, not a
programmer."""
        if content_type == "ai_tips" else
        """You write short, punchy scripts for a "did you know" style
YouTube Shorts channel about surprising true facts (space, science, history,
psychology, nature, animals, or the human body - pick ONE topic at random,
something genuinely surprising and different each time)."""
    )
    topic_rules = topic_rules_ai if content_type == "ai_tips" else topic_rules_random

    return f"""{topic_context}

{avoid_text}

Write the "title" and every scene's "text" in English.

Return ONLY valid JSON in exactly this shape, no extra commentary:
{{
  "title": "a short catchy title for the video, under 8 words",
  "hashtags": ["5 to 8 relevant lowercase hashtags for this specific video, no # symbol"],
  "scenes": [
    {{"text": "one or two spoken sentences", "query": "2-4 word English stock-footage search term for this sentence"}}
  ]
}}

Rules:
{topic_rules}
- Keep language simple, practical, and conversational, suitable for text-to-speech narration.
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


def score_hook(text):
    """Cheap heuristic to rate how scroll-stopping an opening line is. Not a
    substitute for real A/B testing, but a free way to prefer the punchier
    of two AI-generated candidates instead of always taking the first draft."""
    text_l = text.lower()
    score = 0
    if "?" in text:
        score += 2
    if any(ch.isdigit() for ch in text):
        score += 2
    power_words = [
        "secret", "never", "always", "everyone", "nobody", "shocking",
        "mistake", "wrong", "trick", "truth", "surprising", "insane",
        "why", "how", "stop", "most people",
    ]
    score += sum(1 for w in power_words if w in text_l)
    word_count = len(text.split())
    if 6 <= word_count <= 16:  # a hook that's too long or too short lands worse
        score += 1
    return score


def generate_best_script(content_type, candidates=2):
    """Generate a couple of script candidates and keep the one with the
    strongest hook (first scene's opening line), instead of settling for
    whatever comes back on the first try. Costs one extra free-tier Gemini
    call per video."""
    if not GEMINI_KEY:
        return FALLBACK_SCRIPTS[content_type]

    best, best_score = None, -1
    for _ in range(candidates):
        data = generate_script_with_ai(content_type)
        if data is FALLBACK_SCRIPTS[content_type]:
            continue  # a fallback isn't a real candidate to compare
        score = score_hook(data["scenes"][0]["text"])
        if score > best_score:
            best, best_score = data, score

    return best if best is not None else FALLBACK_SCRIPTS[content_type]


async def _edge_tts_save(text, voice, output_path):
    """Save narration audio and, where available, collect per-word timing
    from edge-tts's WordBoundary events (used to sync karaoke-style captions
    exactly to the spoken audio). A bad/missing timing event is skipped
    rather than failing the whole narration."""
    communicate = edge_tts.Communicate(text, voice)
    word_timings = []
    with open(output_path, "wb") as f:
        async for chunk in communicate.stream():
            ctype = chunk.get("type")
            if ctype == "audio":
                f.write(chunk["data"])
            elif ctype == "WordBoundary":
                try:
                    start = chunk["offset"] / 1e7  # 100-ns units -> seconds
                    dur = chunk["duration"] / 1e7
                    word_timings.append({"text": chunk["text"], "start": start, "end": start + dur})
                except Exception:
                    pass  # timing is a nice-to-have; don't let a bad entry break narration
    return word_timings


def synthesize_speech(text, output_path):
    """Generate the narration audio using Microsoft Edge's free neural TTS
    (via the edge-tts library) - no API key, no account, no cost, and much
    more natural-sounding than gTTS. Returns a list of per-word timings
    (empty if unavailable) for karaoke-style captions. Falls back to gTTS
    automatically if edge-tts fails for any reason (no internet, service
    hiccup) - in that case timing is unavailable and captions fall back to
    evenly-spaced chunks, so the pipeline never crashes."""
    try:
        return asyncio.run(_edge_tts_save(text, EDGE_TTS_VOICE, output_path))
    except Exception as e:
        print(f"edge-tts failed, falling back to gTTS: {type(e).__name__}: {e}")

    gTTS(text=text, lang="en").save(output_path)
    return []


def apply_zoom(clip, duration, zoom_ratio=0.15):
    """Subtle Ken Burns style zoom-in over the clip's duration, so the
    background feels alive instead of a static shot. Cheap production-value
    boost, no extra API/cost involved."""
    try:
        def zoom_factor(t):
            return 1 + zoom_ratio * (t / duration)

        zoomed = clip.resized(zoom_factor)
        return CompositeVideoClip(
            [zoomed.with_position("center")], size=(WIDTH, HEIGHT)
        ).with_duration(duration)
    except Exception as e:
        print(f"Zoom effect failed, using plain clip: {e}")
        return clip


def fetch_clip(query, duration_needed, index, used_video_ids):
    """Fetch background footage for a scene. Pulls several results per query
    (not just the first) and splices together 2-3 different clips the video
    hasn't already used yet - much more visually dynamic than looping a
    single shot for the whole scene, and avoids the same footage repeating
    across different scenes. Falls back to a blank clip if Pexels has
    nothing usable."""
    headers = {"Authorization": PEXELS_KEY}
    url = "https://api.pexels.com/videos/search?query=" + query + chr(38) + "per_page=6"

    candidates = []
    for attempt in range(3):
        try:
            time.sleep(1)
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200 and resp.json().get("videos"):
                candidates = resp.json()["videos"]
                break
            else:
                print(f"Pexels returned status {resp.status_code} for query '{query}': {resp.text[:200]}")
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {query}: {e}")
            time.sleep(2)

    if not candidates:
        return ImageClip(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)).with_duration(duration_needed)

    # Prefer clips not already used elsewhere in this video; if the query
    # just doesn't have enough distinct results, allow reuse rather than fail.
    fresh = [v for v in candidates if v["id"] not in used_video_ids] or candidates

    # Split the scene into 2-3 different shots for a faster-paced, more
    # dynamic feel. Skip splitting very short scenes - a cut every <2.5s
    # looks frantic rather than dynamic.
    max_segments = 3 if duration_needed >= 7 else (2 if duration_needed >= 4 else 1)
    chosen = fresh[:min(max_segments, len(fresh))]
    seg_duration = duration_needed / len(chosen)

    segment_clips = []
    for seg_idx, video in enumerate(chosen):
        used_video_ids.add(video["id"])
        video_file = f"bg_{index}_{seg_idx}.mp4"
        try:
            v_files = video["video_files"]
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

            loop_clips = []
            cur_dur = 0
            while cur_dur < seg_duration:
                loop_clips.append(clip)
                cur_dur += clip.duration
            seg_clip = concatenate_videoclips(loop_clips).subclipped(0, seg_duration)
            segment_clips.append(apply_zoom(seg_clip, seg_duration))
        except Exception as e:
            print(f"Could not use clip for '{query}' segment {seg_idx}: {e}")

    if not segment_clips:
        return ImageClip(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)).with_duration(duration_needed)

    combined = concatenate_videoclips(segment_clips)
    if combined.duration < duration_needed:  # rounding can leave a hair short
        last_frame = combined.get_frame(max(combined.duration - 0.04, 0))
        pad = ImageClip(last_frame).with_duration(duration_needed - combined.duration)
        combined = concatenate_videoclips([combined, pad])
    return combined.subclipped(0, duration_needed)


def add_caption(bg_clip, caption_text, duration, word_timings=None, is_cta=False):
    """Render on-screen captions. For normal scenes with word-level timing
    from edge-tts, shows one word at a time exactly synced to the narration
    (karaoke-style) - punchier and far better synced than a fixed-duration
    chunk. Falls back to evenly-spaced 4-word chunks when no timing is
    available (e.g. the gTTS fallback path), and shows the CTA line as one
    static caption for the whole scene. Falls back to the plain background
    clip if rendering fails for any reason, so the pipeline never crashes."""
    try:
        caption_clips = []

        if is_cta:
            txt_clip = TextClip(
                font=FONT_PATH, text=caption_text,
                font_size=72, color="yellow",
                stroke_color="black", stroke_width=3,
                method="caption", size=(int(WIDTH * 0.85), None), text_align="center",
            ).with_duration(duration).with_position(("center", int(HEIGHT * 0.72)))
            caption_clips.append(txt_clip)

        elif word_timings:
            # Karaoke-style: one word at a time, precisely timed to speech.
            for w in word_timings:
                start = max(0, min(w["start"], duration))
                end = max(start, min(w["end"], duration))
                if end <= start:
                    continue
                txt_clip = TextClip(
                    font=FONT_PATH, text=w["text"].upper(),
                    font_size=78, color="white",
                    stroke_color="black", stroke_width=3,
                    method="caption", size=(int(WIDTH * 0.85), None), text_align="center",
                ).with_duration(end - start).with_start(start).with_position(("center", int(HEIGHT * 0.72)))
                caption_clips.append(txt_clip)
            if not caption_clips:
                raise ValueError("No usable word timings")

        else:
            # No timing available - fall back to evenly-spaced word chunks.
            chunk_words = 4
            words = caption_text.split()
            chunks = [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), chunk_words)] or [caption_text]
            chunk_duration = duration / len(chunks)
            for idx, chunk in enumerate(chunks):
                txt_clip = TextClip(
                    font=FONT_PATH, text=chunk,
                    font_size=62, color="white",
                    stroke_color="black", stroke_width=2,
                    method="caption", size=(int(WIDTH * 0.85), None), text_align="center",
                ).with_duration(chunk_duration).with_start(idx * chunk_duration).with_position(("center", int(HEIGHT * 0.72)))
                caption_clips.append(txt_clip)

        return CompositeVideoClip([bg_clip] + caption_clips, size=(WIDTH, HEIGHT)).with_duration(duration)
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


def upload_to_youtube(video_path, title, description, tags, category_id="27"):
    youtube = get_youtube_client()

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": category_id,
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
    target_duration = round(random.uniform(MIN_DURATION, MAX_DURATION), 1)
    content_type = get_run_config()

    script = generate_best_script(content_type)
    title = script["title"]
    scenes = list(script["scenes"]) + [CTA_SCENE]

    print(f"Content type: {content_type} | Topic: {title} | Target duration: {target_duration}s")

    used_video_ids = set()  # avoid the same stock clip showing up twice in one video
    audio_clips = []
    video_clips = []
    for i, scene in enumerate(scenes):
        fname = f"part_{i}.mp3"
        word_timings = synthesize_speech(scene["text"], fname)
        aclip = AudioFileClip(fname)
        audio_clips.append(aclip)

        bg_clip = fetch_clip(scene["query"], aclip.duration, i, used_video_ids)
        is_cta = scene.get("cta", False)
        caption_text = scene.get("caption") or scene["text"]
        scene_clip = add_caption(bg_clip, caption_text, aclip.duration, word_timings=word_timings, is_cta=is_cta)
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

    if content_type == "ai_tips":
        base_tags = ["ai", "aitips", "chatgpt", "productivity", "shorts"]
    else:
        base_tags = ["shorts", "facts", "didyouknow"]

    ai_hashtags = [h.strip().lstrip("#").lower() for h in script.get("hashtags", []) if h.strip()]
    all_tags = list(dict.fromkeys(ai_hashtags + base_tags))  # AI's topic-specific tags first, deduped

    hashtag_line = " ".join(f"#{t}" for t in all_tags[:8])
    subscribe_line = "Subscribe for a new video every single day!"
    description = f"{title}\n\n{subscribe_line}\n\n{hashtag_line}"

    video_id = upload_to_youtube(output_path, title, description, all_tags, category_id=CATEGORY_IDS.get(content_type, "27"))
    update_tracker(video_id, title, content_type)


if __name__ == "__main__":
    generate_video()
