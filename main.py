import os
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, Bidirectional, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

# 1. DATA WRANGLING (CREMA-D ONLY)
def load_cremad_data(base_path='cremad/AudioWAV/'):
    print("Loading CREMA-D dataset...")
    cremad = os.listdir(base_path)
    emotions = []
    paths = []
    
    for file in cremad:
        emotion = file.split('_')[2]
        if emotion == 'SAD':
            emotion = 'sadness'
        elif emotion == 'ANG':
            emotion = 'anger'
        elif emotion == 'DIS':
            emotion = 'disgust'
        elif emotion == 'FEA':
            emotion = 'fear'
        elif emotion == 'HAP':
            emotion = 'happiness'
        elif emotion == 'NEU':
            emotion = 'neutral'
        elif emotion == 'SUR':
            emotion = 'surprise'
        else:
            emotion = 'Unknown'
            
        path = os.path.join(base_path, file)
        emotions.append(emotion)
        paths.append(path)
        
    cremad_df = pd.DataFrame(emotions, columns=['Emotion'])
    cremad_df['Path'] = paths

    return cremad_df

# 2. FEATURE EXTRACTION
def extract_mel_spectrogram(path):
    y, sr = librosa.load(path, duration=3, offset=0.5)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    if S_dB.shape[1] < 130:
        pad_width = 130 - S_dB.shape[1]
        S_dB = np.pad(S_dB, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        S_dB = S_dB[:, :130]
    
    return S_dB

def extract_mfcc(path):
    y, sr = librosa.load(path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    # Pad or truncate to ensure uniform shape (40, 130)
    if mfcc.shape[1] < 130:
        pad_width = 130 - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :130]
    
    return mfcc

# 3. MODEL ARCHITECTURES
def build_mel_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def build_mfcc_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_mfcc_crnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Reshape((model.output_shape[1], model.output_shape[2] * model.output_shape[3])),
        
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(64)),
        Dropout(0.2),
        
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 4. MAIN EXECUTION PIPELINE
def main():
    # 1. Load Data
    df = load_cremad_data('cremad/AudioWAV/')
    
    # 2. Encode Labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(df['Emotion'])
    num_classes = len(encoder.classes_)
    print(f"Classes found: {encoder.classes_}")

    # 3. Extract Features
    print("Extracting Mel Spectrograms...")
    X_mel = np.array([extract_mel_spectrogram(path) for path in df['Path']])
    X_mel = X_mel[..., np.newaxis] # Add channel dimension for CNN

    print("Extracting MFCCs...")
    X_mfcc = np.array([extract_mfcc(path) for path in df['Path']])
    X_mfcc = X_mfcc[..., np.newaxis] # Add channel dimension for CNN

    # 4. Train/Test Splits
    print("Splitting data...")
    X_train_mel, X_test_mel, y_train, y_test = train_test_split(X_mel, y, test_size=0.2, random=42)
    X_train_mfcc, X_test_mfcc, _, _ = train_test_split(X_mfcc, y, test_size=0.2, random=42)

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

    # MODEL 1: Mel Spectrogram CNN
    print("\n--- Training Mel Spectrogram CNN ---")
    mel_cnn = build_mel_cnn(X_train_mel.shape[1:], num_classes)
    mel_history = mel_cnn.fit(
        X_train_mel, y_train, 
        validation_data=(X_test_mel, y_test), 
        epochs=50, batch_size=32, 
        callbacks=[early_stop, reduce_lr], verbose=2
    )
    
    # Evaluate
    y_pred_mel = np.argmax(mel_cnn.predict(X_test_mel), axis=1)
    print("Classification Report: Mel Spectrogram CNN")
    print(classification_report(y_test, y_pred_mel, target_names=encoder.classes_))

    # MODEL 2: MFCC CNN
    print("\n--- Training MFCC CNN ---")
    mfcc_cnn = build_mfcc_cnn(X_train_mfcc.shape[1:], num_classes)
    mfcc_cnn_history = mfcc_cnn.fit(
        X_train_mfcc, y_train, 
        validation_data=(X_test_mfcc, y_test), 
        epochs=50, batch_size=32, 
        callbacks=[early_stop, reduce_lr], verbose=2
    )

    # Evaluate
    y_pred_mfcc_cnn = np.argmax(mfcc_cnn.predict(X_test_mfcc), axis=1)
    print("Classification Report: MFCC CNN")
    print(classification_report(y_test, y_pred_mfcc_cnn, target_names=encoder.classes_))

    # MODEL 3: MFCC CRNN
    print("\n--- Training MFCC CRNN ---")
    mfcc_crnn = build_mfcc_crnn(X_train_mfcc.shape[1:], num_classes)
    mfcc_crnn_history = mfcc_crnn.fit(
        X_train_mfcc, y_train, 
        validation_data=(X_test_mfcc, y_test), 
        epochs=50, batch_size=32, 
        callbacks=[early_stop, reduce_lr], verbose=2
    )

    # Evaluate
    y_pred_mfcc_crnn = np.argmax(mfcc_crnn.predict(X_test_mfcc), axis=1)
    print("Classification Report: MFCC CRNN")
    print(classification_report(y_test, y_pred_mfcc_crnn, target_names=encoder.classes_))
    
    print("\nPipeline Execution Complete!")

if __name__ == "__main__":
    main()