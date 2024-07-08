import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
from proctor import face_module  # Import the proctor function
import random

def load_dataset(dataset_path, sample_size=None, seed=42):
    X = []
    y = []
    id_photo_path = os.path.join(dataset_path, 'ID.png')
    
    if not os.path.exists(id_photo_path):
        print(f"ID photo not found in {dataset_path}. Exiting...")
        return np.array([]), np.array([])
    
    all_frames = []
    for subfolder in ['cheating_frames', 'not_cheating_frames']:
        subfolder_path = os.path.join(dataset_path, subfolder)
        if not os.path.isdir(subfolder_path):
            print(f"{subfolder} folder not found. Skipping...")
            continue
        
        frames = [f for f in os.listdir(subfolder_path) if f.endswith('.jpg')]
        all_frames.extend([(os.path.join(subfolder_path, f), 1 if subfolder == 'cheating_frames' else 0) for f in frames])

    # Shuffle and sample if sample_size is provided
    random.seed(seed)
    if sample_size and sample_size < len(all_frames):
        all_frames = random.sample(all_frames, sample_size)

    for frame_path, label in all_frames:
        proctor_output = face_module(frame_path, id_photo_path)
        
        # Convert proctor output to feature vector
        features = [
            int(proctor_output.get('Identity', False)),
            proctor_output.get('Number of people', 1),
            int(proctor_output.get('Hand Detected', False)),
            int(proctor_output.get('Prohibited Item Use', False)),
            proctor_output.get('Distance', 0),
            proctor_output.get('Illegal Objects', 0),
            #int(proctor_output.get('Prohibited Item', False)),
            encode_face_direction(proctor_output.get('Face Direction', 'center')),
            encode_face_zone(proctor_output.get('Face Zone', 'green')),
            encode_eye_direction(proctor_output.get('Eye Direction', 'center')),
            encode_mouth(proctor_output.get('Mouth', 'closed')),
            #proctor_output.get('Cheat Score', 0)
        ]
        
        X.append(features)
        y.append(label)
    
    return np.array(X), np.array(y)

def encode_face_direction(direction):
    directions = ['left', 'right', 'up', 'down', 'forward']
    return directions.index(direction.lower()) if direction.lower() in directions else len(directions)

def encode_face_zone(zone):
    zones = ['white', 'yellow', 'red']
    return zones.index(zone.lower()) if zone.lower() in zones else len(zones)

def encode_eye_direction(direction):
    directions = ['left', 'right', 'center']
    return directions.index(direction.lower()) if direction.lower() in directions else len(directions)

def encode_mouth(state):
    states = ['GREEN', 'YELLOW', 'RED']
    return states.index(state.lower()) if state.lower() in states else len(states)

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model

def save_model(model, filename):
    dump(model, filename)
    print(f"Model saved as {filename}")

def main():
    dataset_path = "/home/kashyap/Documents/Projects/PROCTOR/anti-cheat-system/Proctor/Dataset"
    X, y = load_dataset(dataset_path, 50)
    
    if len(X) == 0:
        print("No valid data found. Please check your dataset.")
        return
    
    print(f"Dataset loaded. Total samples: {len(X)}")
    
    model = train_model(X, y)
    
    model_filename = "cheating_prediction_model.joblib"
    save_model(model, model_filename)

if __name__ == "__main__":
    main()