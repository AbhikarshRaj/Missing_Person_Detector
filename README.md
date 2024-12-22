# Missing Person Detection Using CCTV Video and AI (CNN)

This project uses **Computer Vision** and **Deep Learning** techniques to detect a missing person from CCTV video footage. By leveraging **Convolutional Neural Networks (CNN)** and other computer vision models such as **MTCNN** for face detection and **Facenet** for face recognition, the project processes video frames and matches faces to identify the missing person.

The system processes input video and an image of the missing person, extracts frames, and identifies any matching faces, saving the results to an output folder.

## Tech Stack:
- **Backend**: Flask (for the web interface)
- **Computer Vision Models**: OpenCV, MTCNN, Facenet
- **Libraries**: TensorFlow, Keras, scikit-image, numpy, pandas, scipy

## Folder Structure:
├── input/ # Directory for uploading CCTV video files 
├── input_1/ # Directory for uploading the image of the missing person
├── backend/ # Contains all models and algorithms 
├── facenet_model/ # Pre-trained Facenet model for face recognition 
├── mtcnn_model/ # Pre-trained MTCNN model for face detection 
│ └── utils.py # Helper functions for image preprocessing, face extraction, etc. 
├── output/ # Directory where output images and matching frames are saved 
├── app.py # Flask application for running the project
├── requirements.txt # List of Python dependencies └── README.md # Project documentation


## How It Works:
1. **Input**:
   - Upload a CCTV video in the `input/` folder.
   - Upload the missing person's image in the `input_1/` folder.

2. **Processing**:
   - The project extracts frames from the CCTV video using OpenCV.
   - It uses **MTCNN** for face detection and **Facenet** for face recognition to compare faces in each frame with the image of the missing person.
   
3. **Output**:
   - The system saves frames where a match is found in the `output/` folder.
   - The images of the matching faces are also stored in the output folder.

4. **Results**:
   - The matching faces, along with their respective frames, are saved in the `output/` folder. These are images where the missing person has been detected in the CCTV video frames.

## Requirements:

This project requires several Python libraries. You can install them using the provided `requirements.txt`.

### To install the dependencies:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/missing-person-detection.git
   cd missing-person-detection

## pip install -r requirements.txt
1.)Running the Application:
2.)Start the Flask application by running:
3.)python app.py

## The application will start a local server, and you can access it via http://127.0.0.1:5000/ in your browser.

## On the web interface:

-Upload a CCTV video in the input/ folder.
-Upload the missing person's image in the input_1/ folder.
-The system will process the video, detect faces, and match them to the image of the missing person.
-The results (matching frames) will be saved in the output/ folder.
