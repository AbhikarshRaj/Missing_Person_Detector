from mtcnn import MTCNN
import numpy as np

def detect_faces(frames):
    detector = MTCNN()
    face_images = []

    for frame in frames:
        # Ensure frame is in RGB format
        if frame.shape[-1] != 3:  # Check for 3 color channels
            raise ValueError("Frames must be in RGB format.")

        # Detect faces
        results = detector.detect_faces(frame)

        for result in results:
            if 'box' in result:
                x, y, width, height = result['box']

                # Handle potential negative coordinates
                x = max(0, x)
                y = max(0, y)

                # Crop the face region
                face = frame[y:y+height, x:x+width]

                # Ensure valid face dimensions
                if face.size > 0:
                    face_images.append(face)

    return face_images
