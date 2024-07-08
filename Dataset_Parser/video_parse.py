import cv2
import os
from datetime import timedelta

def extract_frames(video_path, output_folder):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_time = timedelta(seconds=frame_count / fps)
        output_path = os.path.join(output_folder, f"frame_{frame_count:06d}_{frame_time}.jpg")
        cv2.imwrite(output_path, frame)
        frame_count += 1

    video.release()
    return fps

def get_cheating_intervals():
    intervals = []
    while True:
        start = input("Enter start time for cheating interval (in seconds) or 'q' to quit: ")
        if start.lower() == 'q':
            break
        end = input("Enter end time for cheating interval (in seconds): ")
        intervals.append((timedelta(seconds=float(start)), timedelta(seconds=float(end))))
    return intervals

def sort_frames(input_folder, cheating_folder, not_cheating_folder, cheating_intervals, fps):
    if not os.path.exists(cheating_folder):
        os.makedirs(cheating_folder)
    if not os.path.exists(not_cheating_folder):
        os.makedirs(not_cheating_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):
            frame_time = timedelta(seconds=int(filename.split("_")[1]) / fps)
            
            is_cheating = any(start <= frame_time < end for start, end in cheating_intervals)
            
            if is_cheating:
                os.rename(os.path.join(input_folder, filename),
                          os.path.join(cheating_folder, filename))
            else:
                os.rename(os.path.join(input_folder, filename),
                          os.path.join(not_cheating_folder, filename))

def main():
    video_path = input("Enter the path to your video file: ")
    output_folder = "extracted_frames"
    cheating_folder = "cheating_frames"
    not_cheating_folder = "not_cheating_frames"

    fps = extract_frames(video_path, output_folder)
    cheating_intervals = get_cheating_intervals()
    sort_frames(output_folder, cheating_folder, not_cheating_folder, cheating_intervals, fps)

    print("Processing complete. Frames have been sorted into 'cheating_frames' and 'not_cheating_frames' folders.")

if __name__ == "__main__":
    main()