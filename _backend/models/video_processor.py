import cv2

def extract_frames(video_path, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    frames=[]
    frame_count = 0

    # Ensure the output folder exists
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
    # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        # Resize frame to match DeepFace's input size (224x224 by default)
        frame = cv2.resize(frame, (224,224))
        if frame_count % frame_interval == 0:  # Extract every nth frame
            # Save the frame as an image file
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames
