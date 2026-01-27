import os
import pickle
import json
from deepface import DeepFace

DATASET_PATH = "dataset"
EMBEDDINGS_FILE = "embeddings.pkl"
STATUS_FILE = "training_status.json"

def update_status(message, progress, processed, total):
    """Update training status file"""
    status = {
        "is_training": True,
        "progress": progress,
        "total_students": total,
        "processed_students": processed,
        "message": message,
        "error": None
    }
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f)

def train_model():
    print("Starting training...")
    
    # Count total students first
    total_students = 0
    for batch in os.listdir(DATASET_PATH):
        batch_path = os.path.join(DATASET_PATH, batch)
        if not os.path.isdir(batch_path):
            continue
            
        for dept in os.listdir(batch_path):
            dept_path = os.path.join(batch_path, dept)
            if not os.path.isdir(dept_path):
                continue
                
            for student in os.listdir(dept_path):
                student_path = os.path.join(dept_path, student)
                if os.path.isdir(student_path):
                    total_students += 1
    
    if total_students == 0:
        print("No students found for training!")
        return
    
    embeddings = []
    processed = 0
    
    for batch in os.listdir(DATASET_PATH):
        batch_path = os.path.join(DATASET_PATH, batch)
        if not os.path.isdir(batch_path):
            continue

        for dept in os.listdir(batch_path):
            dept_path = os.path.join(batch_path, dept)
            if not os.path.isdir(dept_path):
                continue

            for student in os.listdir(dept_path):
                student_path = os.path.join(dept_path, student)
                if not os.path.isdir(student_path):
                    continue

                # student folder name: roll_name
                try:
                    roll, name = student.split("_", 1)
                except:
                    roll, name = student, "Unknown"

                print(f"Processing: {roll} - {name}")
                update_status(f"Processing: {roll} - {name}", 
                            int((processed / total_students) * 100),
                            processed, total_students)

                for img in os.listdir(student_path):
                    if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(student_path, img)

                        try:
                            embedding = DeepFace.represent(
                                img_path=img_path,
                                model_name="Facenet",
                                enforce_detection=False,
                                detector_backend="retinaface"
                            )[0]["embedding"]

                            embeddings.append({
                                "roll": roll,
                                "name": name,
                                "dept": dept,
                                "batch": batch,
                                "embedding": embedding
                            })

                        except Exception as e:
                            print(f"Error processing {img_path}: {e}")

                processed += 1

    # Save embeddings
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(embeddings, f)

    # Final status
    update_status(f"Training completed! Processed {len(embeddings)} face samples", 
                  100, processed, total_students)
    
    print("Training completed successfully!")
    print(f"Total face samples stored: {len(embeddings)}")
    print(f"Embeddings saved to {EMBEDDINGS_FILE}")

if __name__ == "__main__":
    train_model()