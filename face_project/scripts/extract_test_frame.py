import cv2
import os

video_path = "face_project/videos/PERSON1_4SN23ADXXXX.mp4"
output_path = "face_project/test_frame_shafiq.jpg"

if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        # Save the 10th frame to avoid black start
        for _ in range(10):
            ret, frame = cap.read()
        cv2.imwrite(output_path, frame)
        print(f"Saved test frame to {output_path}")
    cap.release()
else:
    print("Video not found")
