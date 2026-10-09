import subprocess
import os

img_dir = r"C:\Users\alexander\.gemini\antigravity\scratch\quantforge\docs\images"
frames = [
    os.path.join(img_dir, "quantforge_dashboard.png"),
    os.path.join(img_dir, "quantforge_pipelines.png"),
    os.path.join(img_dir, "quantforge_studio.png"),
    os.path.join(img_dir, "quantforge_eval.png"),
    os.path.join(img_dir, "quantforge_models.png"),
]

concat_file = os.path.join(img_dir, "concat.txt")
with open(concat_file, "w", encoding="utf-8") as f:
    for frame in frames:
        escaped = frame.replace("\\", "/")
        f.write(f"file '{escaped}'\nduration 2\n")
    escaped_last = frames[-1].replace("\\", "/")
    f.write(f"file '{escaped_last}'\n")

out_gif = os.path.join(img_dir, "quantforge_demo.gif")
palette = os.path.join(img_dir, "palette.png")

subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file, "-vf", "scale=1280:-1:flags=lanczos,palettegen", palette], check=True)
subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file, "-i", palette, "-lavfi", "scale=1280:-1:flags=lanczos [x]; [x][1:v] paletteuse", out_gif], check=True)

if os.path.exists(palette):
    os.remove(palette)
if os.path.exists(concat_file):
    os.remove(concat_file)

print("Generated demo GIF successfully:", out_gif, os.path.getsize(out_gif), "bytes")
