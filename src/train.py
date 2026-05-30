"""
Module: train.py
Description: Main code for training the model
"""

import os
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

from data_loader import LipReadingDataLoader
from models_non_public import SequenceModelFactory

def plot_training_history(history, model_name, ref_point, save_dir, date_str):
    plt.figure(figsize=(10, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Accuracy per Epoch - {model_name}', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Loss per Epoch - {model_name}', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"{date_str}-acc_loss-{model_name}-{ref_point}.png")
    plt.savefig(plot_path)
    print(f"[INFO] Training graph is saved at: {plot_path}")
    plt.close()

def main(args):
    # Setup Directory path
    tanggal = datetime.now().strftime('%Y-%m-%d-%H%M')
    model_dir = "models"
    result_dir = os.path.join("results", f"{tanggal}-{args.reference}-{args.model}")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # 1. Load Data
    loader = LipReadingDataLoader(base_folder=args.data_path, batch_size=args.batch_size)
    X, Y, class_weights = loader.load_and_preprocess(args.vector_type, args.reference)
    
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    # 2. Build and compile
    factory = SequenceModelFactory()
    num_classes = Y.shape[1]
    
    print(f"\nBuild the architecture {args.model}...")
    base_model = factory.build_model(args.model, input_shape=X.shape[1:], num_classes=num_classes)
    model = factory.compile_model(base_model)

    # 3. PTraining Process
    print("\nStart training...")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=loader.get_callbacks(),
        verbose=1
    )

    # 4. Simpan Model Akhir
    model_filename = f"{tanggal}-{args.model}-{args.reference}.h5"
    model_path = os.path.join(model_dir, model_filename)
    model.save(model_path)
    print(f"\nModel is saved at: {model_path}")

    # 5. Ekspor Visualisasi
    plot_training_history(history, args.model, args.reference, result_dir, tanggal)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script Training Lip Reading AI")
    parser.add_argument("--data_path", type=str, required=True, help="Data path")
    parser.add_argument("--model", type=str, required=True, help="Type of model (exp: 1DCNN-BIGRU)")
    parser.add_argument("--vector_type", type=str, default="vector", help="vector feature or delta vector feature")
    parser.add_argument("--reference", type=str, required=True, help="Reference: nose, chin, mouth")
    parser.add_argument("--epochs", type=int, default=200, help="Total of epoch")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    
    args = parser.parse_args()
    main(args)