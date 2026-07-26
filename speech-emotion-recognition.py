# Speech Emotion Recognition (SER) using CNNs and CRNNs Based on Mel Spectrograms and Mel Frequency Cepstral Coefficients (MFCCs).

import os
import pandas as pd
import numpy as np
import librosa

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings('ignore')

# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# ==========================================
# 1 & 2- Data Collection and Wrangling
# ==========================================
print("\nLoading datasets...")

# ----------------- CREMA-D -----------------
cremad_dir = 'data/cremad/AudioWAV/'
emotions = []
paths = []

if os.path.exists(cremad_dir):
    for file in os.listdir(cremad_dir):
        if not file.endswith('.wav'):
            continue
        emotion = file.split('_')[2]
        if emotion == 'SAD': emotion = 'sadness'
        elif emotion == 'ANG': emotion = 'anger'
        elif emotion == 'DIS': emotion = 'disgust'
        elif emotion == 'FEA': emotion = 'fear'
        elif emotion == 'HAP': emotion = 'happiness'
        elif emotion == 'NEU': emotion = 'neutral'
        elif emotion == 'SUR': emotion = 'surprise'
        else: emotion = 'Unknown'
        
        path = os.path.join(cremad_dir, file)
        emotions.append(emotion)
        paths.append(path)

cremad_df = pd.DataFrame(emotions, columns=['Emotion'])
cremad_df['Path'] = paths

# ----------------- RAVDESS -----------------
ravdess_dir = 'data/ravdess/'
emotions = []
paths = []

if os.path.exists(ravdess_dir):
    for dir_name in os.listdir(ravdess_dir):
        dir_path = os.path.join(ravdess_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue
        for file in os.listdir(dir_path):
            if not file.endswith('.wav'):
                continue
            emotion = file.split('-')[2]
            if emotion == '01': emotion = 'neutral'
            elif emotion == '02': emotion = 'calm'
            elif emotion == '03': emotion = 'happiness'
            elif emotion == '04': emotion = 'sadness'
            elif emotion == '05': emotion = 'anger'
            elif emotion == '06': emotion = 'fear'
            elif emotion == '07': emotion = 'disgust'
            elif emotion == '08': emotion = 'surprise'
            else: emotion = 'Unknown'
            
            path = os.path.join(dir_path, file)
            emotions.append(emotion)
            paths.append(path)

ravdess_df = pd.DataFrame(emotions, columns=['Emotion'])
ravdess_df['Path'] = paths

# ----------------- TESS -----------------
# Ajustado para acessar a subpasta 'tess' conforme a imagem image_2e71ef.png
tess_dir = 'data/tess/tess/'
emotions = []
paths = []

if os.path.exists(tess_dir):
    for dir_name in os.listdir(tess_dir):
        dir_path = os.path.join(tess_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue
        for file in os.listdir(dir_path):
            if not file.endswith('.wav'):
                continue
            emotion = file.split('.')[0]
            emotion = emotion.split('_')[-1]
            
            if emotion == 'ps': emotion = 'surprise'
            elif emotion == 'sad': emotion = 'sadness'
            elif emotion == 'disgust': emotion = 'disgust'
            elif emotion == 'angry': emotion = 'anger'
            elif emotion == 'happy': emotion = 'happiness'
            elif emotion == 'neutral': emotion = 'neutral'
            elif emotion == 'fear': emotion = 'fear'
            else: emotion = 'Unknown'
            
            path = os.path.join(dir_path, file)
            emotions.append(emotion)
            paths.append(path)

tess_df = pd.DataFrame(emotions, columns=['Emotion'])
tess_df['Path'] = paths

# ----------------- SAVEE -----------------
# Ajustado para acessar os arquivos direto na raiz 'data/savee/' conforme a imagem image_2e71eb.png
savee_dir = 'data/savee/'
emotions = []
paths = []

if os.path.exists(savee_dir):
    for file in os.listdir(savee_dir):
        if not file.endswith('.wav'):
            continue
        
        name_parts = file.split('.')[0].split('_')
        if len(name_parts) < 2:
            continue
        emotion_raw = name_parts[1]
        emotion = emotion_raw[:-2]
        
        if emotion == 'a': emotion = 'anger'
        elif emotion == 'd': emotion = 'disgust'
        elif emotion == 'f': emotion = 'fear'
        elif emotion == 'h': emotion = 'happiness'
        elif emotion == 'n': emotion = 'neutral'
        elif emotion == 'sa': emotion = 'sadness'
        elif emotion == 'su': emotion = 'surprise'
        else: emotion = 'Unknown'
        
        path = os.path.join(savee_dir, file)
        emotions.append(emotion)
        paths.append(path)

savee_df = pd.DataFrame(emotions, columns=['Emotion'])
savee_df['Path'] = paths


# ==========================================
# 3- Data Preparation
# ==========================================
# Combine datasets
df = pd.concat([cremad_df, ravdess_df, tess_df, savee_df], axis=0)
df = df.reset_index(drop=True)

print("\nTotal Emotion Counts (Before dropping 'calm'):")
print(df['Emotion'].value_counts())

# Drop 'calm' emotion
df = df[df['Emotion'] != 'calm']
print("\nEmotion Counts After Dropping 'calm':")
print(df['Emotion'].value_counts())

# ==========================================
# 4- Data Preprocessing
# ==========================================
# Encode the emotion labels into numbers 
encoder = LabelEncoder()
df['Emotion'] = encoder.fit_transform(df['Emotion'])
num_classes = len(encoder.classes_)

print("\nEncoded Emotion Labels Count:")
print(df['Emotion'].value_counts())
print(f"Classes Map: {encoder.classes_}")

# Create output directories for models and results
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# List to store results for CSV
all_results_df = []

# Setup Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, mode='min')
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.000001, mode='min')


# ------------------------------------------
# PROCESS 1: MEL SPECTROGRAMS
# ------------------------------------------
def process_audio_mel(path):
    audio, sr = librosa.load(path, sr=44100, duration=4, mono=True)
    if len(audio) < 4 * sr:
        audio = np.pad(audio, pad_width=(0, 4 * sr - len(audio)), mode='constant')
    signal = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    signal = librosa.power_to_db(signal, ref=np.min)
    return np.array(signal)

print("\nExtracting Mel Spectrograms... (This might take a few minutes)")
X_mel = [process_audio_mel(path) for path in df['Path']]
y_mel = df['Emotion']

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_mel, y_mel, test_size=0.2, random_state=0, shuffle=True)
X_train_1, X_test_1 = np.array(X_train_1), np.array(X_test_1)
y_train_1, y_test_1 = np.array(y_train_1), np.array(y_test_1)

# Standardize
mean_mel, std_mel = np.mean(X_train_1), np.std(X_train_1)
X_train_1 = (X_train_1 - mean_mel) / std_mel
X_test_1 = (X_test_1 - mean_mel) / std_mel

# Reshape
X_train_1 = X_train_1.reshape(X_train_1.shape[0], 128, 345, 1)
X_test_1 = X_test_1.reshape(X_test_1.shape[0], 128, 345, 1)

# Datasets
batch_size = 32
train_dataset_1 = tf.data.Dataset.from_tensor_slices((X_train_1, y_train_1)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
test_dataset_1 = tf.data.Dataset.from_tensor_slices((X_test_1, y_test_1)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# Model 5.1: Mel Spectrogram CNN
print("\n--- Training Mel Spectrogram CNN ---")
model_mel = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 345, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model_mel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_mel.fit(train_dataset_1, epochs=100, validation_data=test_dataset_1, callbacks=[early_stop, lr_scheduler], verbose=2)
model_mel.save('models/emotion_recognition_mel_spec.keras')

y_pred_mel = np.argmax(model_mel.predict(X_test_1), axis=1)
print("\nRESULTS: Mel Spectrogram CNN")
print(classification_report(y_test_1, y_pred_mel, target_names=encoder.classes_))

# Save results for CSV
report_mel = classification_report(y_test_1, y_pred_mel, target_names=encoder.classes_, output_dict=True)
df_res_mel = pd.DataFrame(report_mel).transpose().reset_index().rename(columns={'index': 'Class/Metric'})
df_res_mel.insert(0, 'Model', 'Mel CNN')
all_results_df.append(df_res_mel)

# Free memory
del X_mel, train_dataset_1, test_dataset_1, X_test_1


# ------------------------------------------
# PROCESS 2: MFCCs
# ------------------------------------------
def extract_mfcc(path):
    audio, sr = librosa.load(path, sr=44100, duration=4, mono=True)
    if len(audio) < 4 * sr:
        audio = np.pad(audio, pad_width=(0, 4 * sr - len(audio)), mode='constant')
    signal = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=128)
    return np.array(signal)

print("\nExtracting MFCCs... (This might also take a few minutes)")
X_mfcc = [extract_mfcc(path) for path in df['Path']]
y_mfcc = df['Emotion']

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_mfcc, y_mfcc, test_size=0.2, random_state=0, shuffle=True)
X_train_2, X_test_2 = np.array(X_train_2), np.array(X_test_2)
y_train_2, y_test_2 = np.array(y_train_2), np.array(y_test_2)

# Standardize
mean_mfcc, std_mfcc = np.mean(X_train_2), np.std(X_train_2)
X_train_2 = (X_train_2 - mean_mfcc) / std_mfcc
X_test_2 = (X_test_2 - mean_mfcc) / std_mfcc

# Reshape
X_train_2 = X_train_2.reshape(X_train_2.shape[0], 128, 345, 1)
X_test_2 = X_test_2.reshape(X_test_2.shape[0], 128, 345, 1)

train_dataset_2 = tf.data.Dataset.from_tensor_slices((X_train_2, y_train_2)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
test_dataset_2 = tf.data.Dataset.from_tensor_slices((X_test_2, y_test_2)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# Model 5.2: MFCCs CNN Model
print("\n--- Training MFCC CNN ---")
model_mfcc = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 345, 1), padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model_mfcc.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_mfcc.fit(train_dataset_2, epochs=100, validation_data=test_dataset_2, callbacks=[early_stop, lr_scheduler], verbose=2)
model_mfcc.save('models/emotion_recognition_mfcc.keras')

y_pred_mfcc = np.argmax(model_mfcc.predict(X_test_2), axis=1)
print("\nRESULTS: MFCC CNN")
print(classification_report(y_test_2, y_pred_mfcc, target_names=encoder.classes_))

# Save results for CSV
report_mfcc = classification_report(y_test_2, y_pred_mfcc, target_names=encoder.classes_, output_dict=True)
df_res_mfcc = pd.DataFrame(report_mfcc).transpose().reset_index().rename(columns={'index': 'Class/Metric'})
df_res_mfcc.insert(0, 'Model', 'MFCC CNN')
all_results_df.append(df_res_mfcc)


# Model 5.3: MFCCs CRNN Model
print("\n--- Training MFCC CRNN ---")
model_crnn = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(128, 345, 1), padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Reshape((1, 128)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model_crnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_crnn.fit(train_dataset_2, epochs=100, validation_data=test_dataset_2, callbacks=[early_stop, lr_scheduler], verbose=2)
model_crnn.save('models/emotion_recognition_crnn.keras')

y_pred_crnn = np.argmax(model_crnn.predict(X_test_2), axis=1)
print("\nRESULTS: MFCC CRNN")
print(classification_report(y_test_2, y_pred_crnn, target_names=encoder.classes_))

# Save results for CSV
report_crnn = classification_report(y_test_2, y_pred_crnn, target_names=encoder.classes_, output_dict=True)
df_res_crnn = pd.DataFrame(report_crnn).transpose().reset_index().rename(columns={'index': 'Class/Metric'})
df_res_crnn.insert(0, 'Model', 'MFCC CRNN')
all_results_df.append(df_res_crnn)

# Concatenate all results and save to CSV
final_results = pd.concat(all_results_df, ignore_index=True)
final_results.to_csv('results/results.csv', index=False)
print("\nMetrics successfully saved to results/results.csv")

print("\nPipeline Execution Complete!")