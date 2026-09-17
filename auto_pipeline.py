import os
import time
import requests
import numpy as np
import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"): PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS
from gtts import gTTS
from moviepy import VideoFileClip, AudioFileClip, ImageClip, concatenate_videoclips
WIDTH, HEIGHT = 1080, 1920
PEXELS_KEY = os.environ.get("PEXELS_API_KEY")
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
                clip = clip.cropped(x1=x1, y1=0, x2=x1+WIDTH, y2=HEIGHT)
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
        vclip = fetch_clip(query, aclip.duration, i)
        video_clips.append(vclip)
    
    from moviepy import concatenate_audioclips
    final_audio = concatenate_audioclips(audio_clips)
    final_video = concatenate_videoclips(video_clips)
    
    final_clips_list = []
    total_dur = 0
    target_dur = max(final_audio.duration, 60.0)
    while total_dur < target_dur:
        for v in video_clips:
            final_clips_list.append(v)
            total_dur += v.duration
            if total_dur >= target_dur: break
    
    final_video = concatenate_videoclips(final_clips_list).subclipped(0, target_dur)
    final_audio = final_audio.with_duration(target_dur)
    
    final = final_video.with_audio(final_audio)
    final.write_videofile("final_short.mp4", fps=30, codec="libx264", audio_codec="aac", bitrate="5000k")
if __name__ == "__main__": generate_video()