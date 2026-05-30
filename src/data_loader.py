"""
Module: data_loader.py
Description: Load keypoint data.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau
from typing import Tuple, Dict

class LipReadingDataLoader:
    def __init__(self, base_folder: str, target_frames: int = 44, batch_size: int = 32):
        
        self.base_folder = base_folder
        self.target_frames = target_frames
        self.batch_size = batch_size
        self.label_encoder = LabelEncoder()
        
        # Daftar kelas resmi sesuai standar dataset
        self.valid_classes = [
            'air', 'aku', 'bakso', 'dia', 'doa', 'kamu', 'kopi', 
            'makan', 'minum', 'novel', 'puding', 'rumus', 'saya', 'sura'
        ]

    def load_and_preprocess(self, vector_type: str, reference: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        
        data_path = os.path.join(self.base_folder, vector_type, reference)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Directory is not found: {data_path}")

        X_raw, Y_raw = [], []
        
        print(f"=== Load Data: {vector_type} | Reference: {reference} ===")
        
        for class_name in self.valid_classes:
            class_path = os.path.join(data_path, class_name)
            if not os.path.isdir(class_path):
                print(f"[WARNING] Class directory is not found: {class_path}")
                continue

            for filename in os.listdir(class_path):
                if filename.endswith('.npy'):
                    file_path = os.path.join(class_path, filename)
                    data = np.load(file_path)
                    
                    # Seleksi jumlah frame minimum
                    if data.shape[0] >= self.target_frames:
                        data_trimmed = data[:self.target_frames]
                        X_raw.append(data_trimmed)
                        Y_raw.append(class_name)

        X = np.array(X_raw)
        Y = np.array(Y_raw)
        print(f"Data is loaded: X={X.shape}, Y={Y.shape}")

        
        X = X[:, :, :, [4, 5, 6, 7]]
        
        
        X_norm = np.zeros_like(X)
        
        for i in range(X.shape[0]):
        
            vx_vy = X[i, :, :, :2].reshape(-1, 2)
            mag = X[i, :, :, 2]
            
            scaler = StandardScaler()
            vx_vy_scaled = scaler.fit_transform(vx_vy)
            
            
            mag_reshaped = mag.reshape(-1, 1) if mag.ndim == 1 else mag
            mag_scaled = scaler.fit_transform(mag_reshaped)
            if mag.ndim == 1:
                mag_scaled = mag_scaled.flatten()
            
            
            vx_scaled = vx_vy_scaled[:, 0].reshape(X.shape[1], X.shape[2])
            vy_scaled = vx_vy_scaled[:, 1].reshape(X.shape[1], X.shape[2])
            
            X_norm[i, :, :, 0] = vx_scaled
            X_norm[i, :, :, 1] = vy_scaled
            X_norm[i, :, :, 2] = mag_scaled
            X_norm[i, :, :, 3] = X[i, :, :, 3] 

        
        X_reshaped = X_norm.reshape(X_norm.shape[0], self.target_frames, -1)
        print(f"X_reshaped: {X_reshaped.shape}")

        
        Y_encoded = self.label_encoder.fit_transform(Y)
        Y_binary = to_categorical(Y_encoded)
        
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(Y_encoded),
            y=Y_encoded
        )
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}

        return X_reshaped, Y_binary, class_weight_dict

    def split_and_create_datasets(self, X: np.ndarray, Y: np.ndarray) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42, stratify=Y
        )
        
        print(f"Distribution -> Train: {len(X_train)} | Test: {len(X_test)}")

        
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024, seed=42).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
        test_dataset = test_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return train_dataset, test_dataset

    @staticmethod
    def get_callbacks() -> list:
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
        )
        return [reduce_lr]


if __name__ == "__main__":
    BASE_DIR = "data"
    
    loader = LipReadingDataLoader(base_folder=BASE_DIR, target_frames=44, batch_size=32)