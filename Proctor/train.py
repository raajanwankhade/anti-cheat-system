import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from joblib import dump
from proctor import face_module
import torch
import random
import mediapipe as mp
from tqdm import tqdm
import warnings
import logging
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('mediapipe').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

def load_dataset(dataset_path, sample_size=None, seed=42):
    X = []
    y = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/kashyap/Documents/Projects/PROCTOR/YOLO_Weights/best.pt', force_reload=False).to(device)
    mpHands = mp.solutions.hands
    media_pipe_dict = {
        'mpHands': mpHands,
        'hands': mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5),
        'mpdraw': mp.solutions.drawing_utils
    }

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

    random.seed(seed)
    if sample_size and sample_size < len(all_frames):
        all_frames = random.sample(all_frames, sample_size)

    for frame_path, label in tqdm(all_frames, desc="Processing frames", unit="frame"):
        proctor_output = face_module(frame_path, id_photo_path, yolo_model=model, media_pipe=media_pipe_dict)
        features = [
            int(proctor_output.get('Identity', False)),
            proctor_output.get('Number of people', 1),
            int(proctor_output.get('Hand Detected', False)),
            int(proctor_output.get('Prohibited Item Use', False)),
            proctor_output.get('Distance', 0),
            proctor_output.get('Illegal Objects', 0),
            encode_face_direction(proctor_output.get('Face Direction', 'center')),
            encode_face_zone(proctor_output.get('Face Zone', 'green')),
            encode_eye_direction(proctor_output.get('Eye Direction', 'center')),
            encode_mouth(proctor_output.get('Mouth', 'closed')),
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

def print_feature_importances(model, feature_names):
    importances = model.feature_importances_
    for name, importance in zip(feature_names, importances):
        print(f"{name}: {importance}")

def train_and_evaluate_models(X, y):
    feature_names = [
        'Identity',
        'Number of People',
        'Hand Detected',
        'Prohibited Item Use',
        'Distance',
        'Illegal Objects',
        'Face Direction',
        'Face Zone',
        'Eye Direction',
        'Mouth'
    ]

    if len(X) == 0 or len(y) == 0:
        print("Empty input data. Cannot train the models.")
        return None

    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
    print(f"Sample of X (first row): {X[0]}")
    print(f"Sample of y (first 5 elements): {y[:5]}")

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        # 'LightGBM': LGBMClassifier(random_state=42),
        # 'XGBoost': XGBClassifier(random_state=42),
        # 'MLP': MLPClassifier(random_state=42, max_iter=1000),
        # 'SVM': SVC(random_state=42)
    }

    for name, model in models.items():
        print(f"\nTraining and evaluating {name}:")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        # Assuming model is your RandomForestClassifier
        if(name == 'Random Forest'):
            print_feature_importances(model, feature_names)


    return models

def save_models(models, base_filename):
    for name, model in models.items():
        filename = f"{base_filename}_{name.lower().replace(' ', '_')}.joblib"
        dump(model, filename)
        print(f"{name} model saved as {filename}")

def main():
    dataset_path = "/home/kashyap/Documents/Projects/PROCTOR/anti-cheat-system/anti-cheat-system/Dataset_Parser/Dataset"
    X, y = load_dataset(dataset_path, sample_size=300) #max on kshps laptop - 360

    if len(X) == 0:
        print("No valid data found. Please check your dataset.")
        return

    print(f"Dataset loaded. Total samples: {len(X)}")

    models = train_and_evaluate_models(X, y)

    if models:
        save_models(models, "cheating_prediction_model")

if __name__ == "__main__":
    main()
