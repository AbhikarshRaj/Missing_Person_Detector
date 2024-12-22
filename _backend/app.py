import cv2,os
from flask import Flask,jsonify
from models.video_processor import extract_frames
from models.face_detector import detect_faces
from models.face_embedder import generate_embeddings, match_faces
from PIL import Image  # Ensure Image is imported correctly



app = Flask(__name__)

#checks if folder exists if they not it makes
app.config['UPLOAD_FOLDER'] = '../Input/'
app.config['OUTPUT_FOLDER'] = '../Output/'

video_path="../Input/1.mp4"
image_path="../Input_1/1.png"
output_folder="../Output"

# Function to save frames using PIL
def save_frame_with_pil(frame, output_path):
    pil_image = Image.fromarray(frame)
    pil_image.save(output_path)

@app.route('/extract')
def extract_frame():
    try:
     # Step 2: Process Video
        saved_frame = extract_frames(video_path)
        faces_images = detect_faces(saved_frame)

     # Step 3: Generate Embeddings
        input_face = cv2.imread(image_path)
        input_embedding = generate_embeddings([input_face])[0]
        video_embeddings = generate_embeddings(faces_images)

    # Step 4: Match Faces
        match_indices = match_faces(input_embedding, video_embeddings)
        
    # Step 5: Save Matched Frames
        matched_frames = []
        for idx in match_indices:
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"match_{idx}.jpg")
            save_frame_with_pil(saved_frame[idx], output_path)
            matched_frames.append(output_path)
        
    # Step 6: Return Response
        return jsonify({
        "status": "success",
        "matched_frames": matched_frames
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})



if __name__ == "__main__":
    app.run(debug=True)