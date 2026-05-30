"""
Module: predict.py
Description: Model Evaluation (Confusion Matrix & Classification Report).
"""

import os
import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

from data_loader import LipReadingDataLoader

def generate_evaluation_report(Y_true, Y_pred_prob, label_classes):
    Y_pred = np.argmax(Y_pred_prob, axis=1)
    Y_true_idx = np.argmax(Y_true, axis=1)

    print("\n" + "="*50)
    print(" CONFUSION MATRIX ")
    print("="*50)
    cm = confusion_matrix(Y_true_idx, Y_pred)
    print(cm)
    
    print("\n" + "="*50)
    print(" CLASSIFICATION REPORT (dalam %)")
    print("="*50)
    report = classification_report(Y_true_idx, Y_pred, target_names=label_classes, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    
    cols_to_pct = ['precision', 'recall', 'f1-score']
    df_report[cols_to_pct] *= 100
    df_report[cols_to_pct] = df_report[cols_to_pct].round(2)
    print(df_report[['precision', 'recall', 'f1-score', 'support']])

    print("\n" + "="*50)
    print(" ROC AUC SCORES ")
    print("="*50)
    for i, class_name in enumerate(label_classes):
        fpr, tpr, _ = roc_curve(Y_true[:, i], Y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        print(f"Class [{class_name.ljust(8)}] AUC: {roc_auc:.4f}")

def main(args):
    
    loader = LipReadingDataLoader(base_folder=args.data_path)
    X, Y, _ = loader.load_and_preprocess(args.vector_type, args.reference)
    
    
    from sklearn.model_selection import train_test_split
    _, X_test, _, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    
    
    print(f"\nLoad model from: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError("File model (.h5) is not found!")
        
    model = load_model(args.model_path)
    
    
    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print(f"\nAccuracy: {acc:.4f} | Loss: {loss:.4f}")

   
    Y_pred_prob = model.predict(X_test)
    generate_evaluation_report(Y_test, Y_pred_prob, loader.valid_classes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction Script & Lip Reading Evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Complete path to file .h5")
    parser.add_argument("--data_path", type=str, required=True, help="Dataset path")
    parser.add_argument("--vector_type", type=str, default="vector", help="vector or delta_vector")
    parser.add_argument("--reference", type=str, required=True, help="Reference: nose, chin, mouth")
    
    args = parser.parse_args()
    main(args)