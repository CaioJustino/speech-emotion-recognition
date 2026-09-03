# Speech Emotion Recognition (SER) using CNNs and CRNNs Based on Mel Frequency Cepstral Coefficients (MFCCs).

import os
import tensorflow as tf
# 1. ATIVADO: Permite que a GPU descubra o algoritmo mais rápido para as CNNs
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '1' 
# 2. ATIVADO: Liga o compilador XLA para fundir operações matemáticas
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
tf.config.optimizer.set_jit(True) 

import pandas as pd
import numpy as np
import librosa
from joblib import Parallel, delayed
import gc

from tensorflow.keras import mixed_precision
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings('ignore')

# 3. ATIVADO: Mixed Precision para usar os Tensor Cores da Tesla V100
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f"\nCompute dtype: {policy.compute_dtype}")
print(f"Variable dtype: {policy.variable_dtype}\n")

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Configurar o uso da GPU globalmente
strategy = tf.distribute.MirroredStrategy()
print('Num Devices: {}'.format(strategy.num_replicas_in_sync))

# O resto do seu código (coleta de dados, K-Fold, tf.data, etc) continua exatamente igual daqui para baixo...
global_batch_size = 128
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
df = pd.concat([cremad_df, ravdess_df, tess_df, savee_df], axis=0)

# Resetar índice antes e depois do drop para manter a integridade no K-Fold
df = df.reset_index(drop=True)

print("\nTotal Emotion Counts (Before dropping 'calm'):")
print(df['Emotion'].value_counts())

# Drop 'calm' emotion e reseta o índice novamente
df = df[df['Emotion'] != 'calm'].reset_index(drop=True)

print("\nEmotion Counts After Dropping 'calm':")
print(df['Emotion'].value_counts())

# ==========================================
# 4- Data Preprocessing
# ==========================================
encoder = LabelEncoder()
df['Emotion'] = encoder.fit_transform(df['Emotion'])
num_classes = len(encoder.classes_)

print("\nEncoded Emotion Labels Count:")
print(df['Emotion'].value_counts())
print(f"Classes Map: {encoder.classes_}")

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('results/MFCC_CNN', exist_ok=True)
os.makedirs('results/MFCC_CRNN', exist_ok=True)

all_results_df = []

# ==========================================
# FUNÇÕES DE MODELOS E CALLBACKS
# ==========================================
def get_callbacks():
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True, 
        mode='min'
    )
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=3, 
        min_lr=0.000001, 
        mode='min'
    )
    return [early_stop, lr_scheduler]

def build_cnn_model(n_mfcc):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(n_mfcc, 352, 1), padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax') 
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def build_crnn_model(n_mfcc):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(n_mfcc, 352, 1), padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'), 
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Reshape((1, 128)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax') 
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model


# ==========================================
# 5- PROCESS: MFCCs & K-FOLD VALIDATION
# ==========================================

def load_audio_only(path):
    audio, sr = librosa.load(path, sr=44100, duration=4, mono=True)
    if len(audio) < 4 * sr:
        audio = np.pad(audio, pad_width=(0, 4 * sr - len(audio)), mode='constant')
    return audio

print("\nCarregando audios na RAM...")
raw_audios = Parallel(n_jobs=-1)(delayed(load_audio_only)(path) for path in df['Path'])

def extract_mfcc_from_memory(audio_array, sr, n_mfcc_val):
    n_mels_val = max(128, n_mfcc_val)
    signal = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=n_mfcc_val, n_mels=n_mels_val)
    return np.array(signal, dtype=np.float32)

lista_mfccs = [12, 13, 14, 25, 39, 65, 96, 128, 192, 255, 256, 257]
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for n_mfcc in lista_mfccs:
    print(f"\n=======================================================")
    print(f" Executando extração e validação 10-Fold para n_mfcc = {n_mfcc}")
    print(f"=======================================================")

    print(f"Extracting MFCCs ({n_mfcc}) da Memória RAM...")
    X_mfcc = Parallel(n_jobs=-1)(delayed(extract_mfcc_from_memory)(audio, 44100, n_mfcc) for audio in raw_audios)
    X_mfcc = np.array(X_mfcc, dtype=np.float32)
    y_mfcc = np.array(df['Emotion'], dtype=np.int32)

    cnn_true_labels, cnn_pred_labels = [], []
    crnn_true_labels, crnn_pred_labels = [], []

    for fold, (train_index, test_index) in enumerate(skf.split(X_mfcc, y_mfcc), 1):
        
        # Limpar sessão ANTES do treino para que a GPU não misture pesos antigos
        tf.keras.backend.clear_session()
        gc.collect()
        
        print(f"\n--- Fold {fold}/10 (n_mfcc={n_mfcc}) ---")
        
        # Split com cast de segurança
        X_train = np.array(X_mfcc[train_index], dtype=np.float32)
        X_test = np.array(X_mfcc[test_index], dtype=np.float32)
        y_train = np.array(y_mfcc[train_index], dtype=np.int32)
        y_test = np.array(y_mfcc[test_index], dtype=np.int32)

        # Standardize com Epsilon para evitar divisão por zero
        mean_mfcc = np.mean(X_train, dtype=np.float32)
        std_mfcc = np.std(X_train, dtype=np.float32)
        
        X_train = ((X_train - mean_mfcc) / (std_mfcc + 1e-8)).astype(np.float32)
        X_test = ((X_test - mean_mfcc) / (std_mfcc + 1e-8)).astype(np.float32)

        # Pad e Reshape
        pad_width = 352 - 345
        X_train = np.pad(X_train, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
        X_test = np.pad(X_test, ((0, 0), (0, 0), (0, pad_width)), mode='constant')

        X_train = X_train.reshape(X_train.shape[0], n_mfcc, 352, 1).astype(np.float32)
        X_test = X_test.reshape(X_test.shape[0], n_mfcc, 352, 1).astype(np.float32)

        # ===================================================================
        # Otimização de I/O com tf.data.Dataset
        # ===================================================================
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.cache() \
                                     .shuffle(buffer_size=len(X_train)) \
                                     .batch(global_batch_size) \
                                     .prefetch(tf.data.AUTOTUNE)
                                     
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_dataset = test_dataset.cache() \
                                   .batch(global_batch_size) \
                                   .prefetch(tf.data.AUTOTUNE)

        # ---------------- 1. Treinamento MFCC CNN ----------------
        print(f"\nTreinando MFCC CNN - Fold {fold}...")
        with strategy.scope():
            model_mfcc = build_cnn_model(n_mfcc)
        
        model_mfcc.fit(
            train_dataset, 
            epochs=100, 
            validation_data=test_dataset, 
            callbacks=get_callbacks(), 
            verbose=2
        )
        
        y_pred_mfcc = np.argmax(model_mfcc.predict(X_test, verbose=0), axis=1)
        cnn_true_labels.extend(y_test)
        cnn_pred_labels.extend(y_pred_mfcc)
        
        print(f"\nResultados Parciais: CNN (Fold {fold})")
        print(classification_report(y_test, y_pred_mfcc, target_names=encoder.classes_))
        
        if fold == 10:
            model_mfcc.save(f'models/emotion_recognition_mfcc_cnn_{n_mfcc}.keras')

        # Deletar explicitamente antes de invocar o próximo (segurança de memória)
        del model_mfcc
        gc.collect()

        # ---------------- 2. Treinamento MFCC CRNN ----------------
        print(f"\nTreinando MFCC CRNN - Fold {fold}...")
        with strategy.scope():
            model_crnn = build_crnn_model(n_mfcc)

        model_crnn.fit(
            train_dataset, 
            epochs=100, 
            validation_data=test_dataset, 
            callbacks=get_callbacks(), 
            verbose=2
        )
        
        y_pred_crnn = np.argmax(model_crnn.predict(X_test, verbose=0), axis=1)
        crnn_true_labels.extend(y_test)
        crnn_pred_labels.extend(y_pred_crnn)

        print(f"\nResultados Parciais: CRNN (Fold {fold})")
        print(classification_report(y_test, y_pred_crnn, target_names=encoder.classes_))
        
        if fold == 10:
            model_crnn.save(f'models/emotion_recognition_mfcc_crnn_{n_mfcc}.keras')

        # Limpeza pesada ao final do processamento do fold
        del X_train, X_test, y_train, y_test, model_crnn, train_dataset, test_dataset
        gc.collect()

    # ==========================================
    # Consolidação dos Resultados para o N_MFCC atual (Após os 10 folds)
    # ==========================================
    
    print(f"\n=======================================================")
    print(f" RESULTADOS CONSOLIDADOS (10-Fold) PARA N_MFCC = {n_mfcc}")
    print(f"=======================================================")
    
    print(f"\nCONSOLIDADO: MFCC CNN")
    print(classification_report(cnn_true_labels, cnn_pred_labels, target_names=encoder.classes_))
    report_cnn = classification_report(cnn_true_labels, cnn_pred_labels, target_names=encoder.classes_, output_dict=True)
    df_res_cnn = pd.DataFrame(report_cnn).transpose().reset_index().rename(columns={'index': 'Class/Metric'})
    df_res_cnn.insert(0, 'Model', f'MFCC CNN ({n_mfcc})')
    df_res_cnn.to_csv(f'results/MFCC_CNN/{n_mfcc}.csv', index=False)
    all_results_df.append(df_res_cnn)

    print(f"\nCONSOLIDADO: MFCC CRNN")
    print(classification_report(crnn_true_labels, crnn_pred_labels, target_names=encoder.classes_))
    report_crnn = classification_report(crnn_true_labels, crnn_pred_labels, target_names=encoder.classes_, output_dict=True)
    df_res_crnn = pd.DataFrame(report_crnn).transpose().reset_index().rename(columns={'index': 'Class/Metric'})
    df_res_crnn.insert(0, 'Model', f'MFCC CRNN ({n_mfcc})')
    df_res_crnn.to_csv(f'results/MFCC_CRNN/{n_mfcc}.csv', index=False)
    all_results_df.append(df_res_crnn)

    del X_mfcc, y_mfcc
    gc.collect()

# ==========================================
# Finalização
# ==========================================
final_results = pd.concat(all_results_df, ignore_index=True)
final_results.to_csv('results/results_consolidated.csv', index=False)
print("\nMetrics successfully saved to individual folders and consolidated in results/results_consolidated.csv")
print("\nPipeline Execution Complete!")
