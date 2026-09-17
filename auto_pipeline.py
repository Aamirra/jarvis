import os
import time
import requests
import numpy as np
import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"): PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS
from gtts import gTTS
from moviepy import (
    VideoFileClip, AudioFileClip, ImageClip, concatenate_videoclips,
    concatenate_audioclips, TextClip, CompositeVideoClip
)

WIDTH, HEIGHT = 1080, 1920
MIN_DURATION = 60.0  # minimum final video length in seconds
PEXELS_KEY = os.environ.get("PEXELS_API_KEY")

# Font used for on-screen captions (installed via apt in the workflow: fonts-dejavu-core)
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


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
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {query}: {e}")
            time.sleep(2)

    return ImageClip(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)).with_duration(duration_needed)


def add_caption(bg_clip, text, duration):
    """Overlay on-screen caption text on top of a background clip. Falls back
    to the plain background clip if caption rendering fails for any reason,
    so the pipeline never crashes just because of a font/text issue."""
    try:
        caption = TextClip(
            font=FONT_PATH,
            text=text,
            font_size=58,
            color="white",
            stroke_color="black",
            stroke_width=2,
            method="caption",
            size=(int(WIDTH * 0.85), None),
            text_align="center",
        ).with_duration(duration).with_position(("center", int(HEIGHT * 0.72)))

        return CompositeVideoClip([bg_clip, caption], size=(WIDTH, HEIGHT)).with_duration(duration)
    except Exception as e:
        print(f"Caption failed for text '{text[:30]}...': {e}")
        return bg_clip


def generate_video():
    scenes = [
        ("Did you know that space is completely silent?", "deep space cosmos silence"),
        ("Sound waves require a medium like air to travel, and because space is a vacuum, molecules are too far apart to carry sound.", "space vacuum stars"),
        ("If you screamed in space, no one would hear you.", "astronaut floating space"),
        ("This eerie silence stretches across the entire universe, making the cosmos both breathtaking and strangely terrifying.", "galaxy spinning nebula"),
        ("Planets, stars, and galaxies move in complete quiet, hidden behind the vastness of interstellar dark matter.", "planet orbiting space dark")
    ]

    audio_clips = []
    video_clips = []
    for i, (text, query) in enumerate(scenes):
        fname = f"part_{i}.mp3"
        gTTS(text=text, lang="en").save(fname)
        aclip = AudioFileClip(fname)
        audio_clips.append(aclip)

        bg_clip = fetch_clip(query, aclip.duration, i)
        scene_clip = add_caption(bg_clip, text, aclip.duration)
        video_clips.append(scene_clip)

    # Base story (single pass through all scenes, audio+video already in sync)
    story_audio_clips = list(audio_clips)
    story_video_clips = list(video_clips)
    base_duration = sum(a.duration for a in audio_clips)
    total_dur = base_duration

    # If the story is shorter than MIN_DURATION, repeat the WHOLE story
    # (audio + video together) so audio and video always stay in sync.
    # This avoids the old bug where only the video was looped while the
    # audio's duration was force-extended without real sound, causing
    # silent/broken audio in the extra part of the video.
    while total_dur < MIN_DURATION:
        story_audio_clips += audio_clips
        story_video_clips += video_clips
        total_dur += base_duration

    final_audio = concatenate_audioclips(story_audio_clips)
    final_video = concatenate_videoclips(story_video_clips)

    # Trim both to the exact same length just in case of tiny float mismatches
    final_duration = min(final_audio.duration, final_video.duration)
    final_audio = final_audio.subclipped(0, final_duration)
    final_video = final_video.subclipped(0, final_duration)

    final = final_video.with_audio(final_audio)
    final.write_videofile("final_short.mp4", fps=30, codec="libx264", audio_codec="aac", bitrate="5000k")


if __name__ == "__main__":
    generate_video()
