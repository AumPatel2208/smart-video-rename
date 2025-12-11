import os
import subprocess
import requests
from PIL import Image
import base64

# Paths (change these as needed)

# Original and processed paths
video_path = "C0074.MP4"  # Path to your video file
downscaled_path = "downscaled_360p.mp4"  # Temp file for 360p video
frames_dir = "output_frames"
audio_path = "output_audio.aac"


# 1. Downscale video to 360p
subprocess.run([
    "ffmpeg", "-i", video_path, "-vf", "scale=-2:360", "-c:v", "libx264", "-preset", "fast", "-y", downscaled_path
], check=True)

# 2. Extract frames (1 frame per second for demo) from downscaled video
os.makedirs(frames_dir, exist_ok=True)
subprocess.run([
    "ffmpeg", "-i", downscaled_path, "-vf", "fps=1", f"{frames_dir}/frame_%04d.jpg"
], check=True)

# 3. Extract audio from downscaled video
subprocess.run([
    "ffmpeg", "-i", downscaled_path, "-vn", "-acodec", "aac", "-y", audio_path
], check=True)

# 3. Load frames as base64
images = []
for filename in sorted(os.listdir(frames_dir)):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        with open(os.path.join(frames_dir, filename), "rb") as f:
            img_bytes = f.read()
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            images.append(img_b64)

# 4. Prepare content for Ollama
content = []
for img_b64 in images:
    content.append({"type": "image", "content": img_b64})  # base64 string

# Add audio (as base64)
with open(audio_path, "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode("utf-8")
content.append({"type": "audio", "audio": audio_b64})

# Add prompt
content.append({"type": "text", "text": "Describe the events unfolding in this video, including any sounds."})

messages = [{"role": "user", "content": content}]


# 5. Send to Ollama API using the ollama Python package
from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(model='gemma3', messages=messages)
print(response['message']['content'])
# or access fields directly from the response object
print(response.message.content)

# --- Cleanup temp files ---
import shutil

# Remove downscaled video
if os.path.exists(downscaled_path):
    os.remove(downscaled_path)

# Remove audio file
if os.path.exists(audio_path):
    os.remove(audio_path)

# Remove frames directory and its contents
if os.path.exists(frames_dir):
    shutil.rmtree(frames_dir)
