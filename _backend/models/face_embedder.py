from deepface import DeepFace
from scipy.spatial.distance import cosine

def generate_embeddings(face_images):
    embeddings = []
    for face in face_images:
        # Generate the embedding for each face
        representation = DeepFace.represent(face, model_name="Facenet", enforce_detection=False)
        if representation:
            # Extract the embedding from the response
            embeddings.append(representation[0]["embedding"])
    return embeddings

def match_faces(input_embedding, video_embeddings, threshold=0.75 ):
    matches = []
    for idx, embedding in enumerate(video_embeddings):
        # Calculate cosine similarity
        similarity = 1 - cosine(input_embedding, embedding)
        if similarity > threshold:
            matches.append(idx)
    return matches
