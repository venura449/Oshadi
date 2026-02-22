import os
import cv2

video_path = "cctv2.mp4"
output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Cannot open video")
    raise SystemExit

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # fallback if fps not detected

save_interval_sec = 10
frames_per_10sec = int(fps * save_interval_sec)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frames_per_10sec == 0:
        timestamp_sec = frame_count / fps
        filename = f"frame_{frame_count}_t{int(timestamp_sec)}s.jpg"
        cv2.imwrite(os.path.join(output_dir, filename), frame)

    frame_count += 1

cap.release()
print("✅ Frames extracted every 10 seconds successfully")
