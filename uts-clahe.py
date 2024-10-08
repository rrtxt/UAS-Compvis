import cv2
import os

# CLAHE function to enhance contrast of a single frame (image)
def apply_clahe_to_frame(frame):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Create a CLAHE object with clipLimit=2.0 and grid size 8x8
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    # Apply CLAHE to the grayscale frame
    clahe_frame = clahe.apply(gray_frame)
    
    return clahe_frame

# Function to process video and apply CLAHE to each frame
def process_video_with_clahe(input_video, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    video_path = input_video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply CLAHE to each frame
        clahe_frame = apply_clahe_to_frame(frame)
        
        # Create the filename for the CLAHE-processed frame
        frame_filename = f"{output_folder}/clahe_frame_{frame_count:04d}.png"
        cv2.imwrite(frame_filename, clahe_frame)
        
        frame_count += 1
    
    cap.release()

# Function to process all videos in a folder and apply CLAHE to each frame
def process_all_videos_in_folder_with_clahe(input_folder, category_folder, output_folder):
    category_output_folder = os.path.join(output_folder, category_folder)
    
    for video_filename in os.listdir(input_folder):
        if video_filename.endswith(('.mp4', '.avi', '.mkv')):
            video_path = os.path.join(input_folder, video_filename)
            
            video_name = os.path.splitext(video_filename)[0]
            output_video_folder = os.path.join(category_output_folder, video_name)
            
            print(f"Processing {video_filename} with CLAHE")
            process_video_with_clahe(video_path, output_video_folder)
            print(f"CLAHE processed frames saved to {output_video_folder}")

# Example usage
input_folders = {
    r'D:\Kuliah\Semester 7\COMPVIS\UCF50\YoYo': 'YoYo',
    r'D:\Kuliah\Semester 7\COMPVIS\UCF50\PlayingGuitar': 'PlayingGuitar'
}
output_folder = 'output/clahe_video'

for input_folder, category_folder in input_folders.items():
    process_all_videos_in_folder_with_clahe(input_folder, category_folder, output_folder)

print("All videos processed with CLAHE successfully!")
