import os
import time
import requests
import numpy as np
import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"): PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS
from gtts import gTTS
from moviepy import (
    VideoFileClip, AudioFileClip, ImageClip, concatenate_videoclips,
    concatenate_audioclips, TextClip, CompositeVideoClip, AudioClip
)

WIDTH, HEIGHT = 1080, 1920
TARGET_DURATION = 60.0  # final video will always be exactly this long
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
        # Content ran long: trim down to exactly TARGET_DURATION
        final_audio = final_audio.subclipped(0, TARGET_DURATION)
        final_video = final_video.subclipped(0, TARGET_DURATION)
    elif current_duration < TARGET_DURATION:
        # Content ran short: pad with silence + a frozen last frame instead of
        # repeating the whole story, so there is no jarring repeat and no
        # silent/broken audio.
        pad = TARGET_DURATION - current_duration
        silence = AudioClip(lambda t: 0, duration=pad, fps=44100)
        final_audio = concatenate_audioclips([final_audio, silence])

        last_frame = final_video.get_frame(max(final_video.duration - 0.04, 0))
        freeze = ImageClip(last_frame).with_duration(pad)
        final_video = concatenate_videoclips([final_video, freeze])

    # Final safety trim so both are exactly equal length
    final_duration = min(final_audio.duration, final_video.duration)
    final_audio = final_audio.subclipped(0, final_duration)
    final_video = final_video.subclipped(0, final_duration)

    final = final_video.with_audio(final_audio)
    final.write_videofile("final_short.mp4", fps=30, codec="libx264", audio_codec="aac", bitrate="5000k")


if __name__ == "__main__":
    generate_video()
