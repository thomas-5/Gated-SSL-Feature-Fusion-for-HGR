import os
import json
import yaml
import joblib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any

# Reuse the data loader from training script
import sys
sys.path.append('.')
from models.landmark_based.train_landmark_models import HandLandmarkDataLoader

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


class LandmarkModelEvaluator:
    """Evaluate trained landmark-based models on test data."""
    
    def __init__(self, model_dir: str, landmarks_dir: str, config_path: str):
        self.model_dir = model_dir
        self.landmarks_dir = landmarks_dir
        self.config_path = config_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize data loader
        self.data_loader = HandLandmarkDataLoader(landmarks_dir, self.config['global_settings'])
        
        # Load trained models
        self.models = self.load_trained_models()
        
        self.test_results = {}
    
    def load_trained_models(self) -> Dict[str, Any]:
        """Load all trained models from the model directory."""
        models = {}
        
        for model_name in ['svm', 'random_forest']:
            model_path = os.path.join(self.model_dir, f'{model_name}_model.joblib')
            
            if os.path.exists(model_path):
                try:
                    models[model_name] = joblib.load(model_path)
                    print(f"Loaded {model_name} model from {model_path}")
                except Exception as e:
                    print(f"Error loading {model_name} model: {e}")
            else:
                print(f"Model file not found: {model_path}")
        
        return models
    
    def preprocess_test_features(self, X_test: np.ndarray, model_name: str) -> np.ndarray:
        """Preprocess test features using the saved preprocessors."""
        
        model_data = self.models[model_name]
        
        # Apply scaling if used during training
        if model_data['scaler'] is not None:
            X_test = model_data['scaler'].transform(X_test)
        
        # Apply PCA if used during training
        if model_data['pca'] is not None:
            X_test = model_data['pca'].transform(X_test)
        
        return X_test
    
    def evaluate_model_on_test(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate a single model on test data."""
        
        print(f"\nEvaluating {model_name.upper()} on test set...")
        
        # Get model and preprocessors
        model_data = self.models[model_name]
        model = model_data['model']
        
        # Preprocess features
        X_test_processed = self.preprocess_test_features(X_test, model_name)
        
        # Make predictions
        y_pred = model.predict(X_test_processed)
        
        # Get prediction probabilities if available
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test_processed)
        
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'predictions': y_pred.tolist(),
            'true_labels': y_test.tolist()
        }
        
        # Print results
        print(f"Test Accuracy:  {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall:    {recall:.4f}")
        print(f"Test F1-Score:  {f1:.4f}")
        
        return results
    
    def compare_validation_vs_test(self) -> pd.DataFrame:
        """Compare validation and test performance."""
        
        # Load validation results
        val_results_path = os.path.join(self.model_dir, 'evaluation_results.json')
        
        if not os.path.exists(val_results_path):
            print("Warning: Validation results not found. Skipping comparison.")
            return None
        
        with open(val_results_path, 'r') as f:
            val_results = json.load(f)
        
        # Create comparison dataframe
        comparison_data = []
        
        for model_name in self.test_results.keys():
            if model_name in val_results:
                comparison_data.append({
                    'Model': model_name.upper(),
                    'Val_Accuracy': f"{val_results[model_name]['accuracy']:.4f}",
                    'Test_Accuracy': f"{self.test_results[model_name]['accuracy']:.4f}",
                    'Val_F1': f"{val_results[model_name]['f1_score']:.4f}",
                    'Test_F1': f"{self.test_results[model_name]['f1_score']:.4f}",
                    'Accuracy_Diff': f"{self.test_results[model_name]['accuracy'] - val_results[model_name]['accuracy']:.4f}",
                    'F1_Diff': f"{self.test_results[model_name]['f1_score'] - val_results[model_name]['f1_score']:.4f}"
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n" + "="*100)
        print("VALIDATION vs TEST PERFORMANCE COMPARISON")
        print("="*100)
        print(comparison_df.to_string(index=False))
        print("="*100)
        
        return comparison_df
    
    def save_test_results(self):
        """Save test evaluation results."""
        
        # Save detailed results
        results_path = os.path.join(self.model_dir, 'test_evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        print(f"\nTest results saved: {results_path}")
        
        # Create test summary
        test_summary = []
        for model_name, results in self.test_results.items():
            test_summary.append({
                'Model': model_name.upper(),
                'Test_Accuracy': f"{results['accuracy']:.4f}",
                'Test_Precision': f"{results['precision']:.4f}",
                'Test_Recall': f"{results['recall']:.4f}",
                'Test_F1': f"{results['f1_score']:.4f}"
            })
        
        test_summary_df = pd.DataFrame(test_summary)
        summary_path = os.path.join(self.model_dir, 'test_summary.csv')
        test_summary_df.to_csv(summary_path, index=False)
        
        print(f"\nTest Summary:")
        print(test_summary_df.to_string(index=False))
        print(f"\nTest summary saved: {summary_path}")
    
    def run_evaluation(self):
        """Run complete test evaluation."""
        
        print("="*80)
        print("LANDMARK-BASED MODEL TEST EVALUATION")
        print("="*80)
        print(f"Model directory: {self.model_dir}")
        print(f"Landmarks directory: {self.landmarks_dir}")
        print(f"Available models: {list(self.models.keys())}")
        
        # Load test data
        print("\nLoading test data...")
        X_test, y_test, test_files = self.data_loader.load_split_data('test')
        
        # Encode labels (need to fit on same classes as training)
        # Load the label encoder from training
        train_X, train_y, _ = self.data_loader.load_split_data('train')
        self.data_loader.label_encoder.fit(train_y)
        y_test_encoded = self.data_loader.label_encoder.transform(y_test)
        
        print(f"Test samples: {len(X_test)}")
        print(f"Feature dimension: {X_test.shape[1]}")
        
        # Evaluate each model
        for model_name in self.models.keys():
            try:
                results = self.evaluate_model_on_test(model_name, X_test, y_test_encoded)
                self.test_results[model_name] = results
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                continue
        
        # Save results
        self.save_test_results()
        
        # Compare with validation results
        comparison_df = self.compare_validation_vs_test()
        if comparison_df is not None:
            comp_path = os.path.join(self.model_dir, 'validation_vs_test_comparison.csv')
            comparison_df.to_csv(comp_path, index=False)
            print(f"Comparison saved: {comp_path}")
        
        print(f"\n{'='*80}")
        print("TEST EVALUATION COMPLETED!")
        print(f"{'='*80}")


def main():
    """Main function for test evaluation."""
    
    parser = argparse.ArgumentParser(description='Evaluate trained landmark-based models on test set')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing trained models')
    parser.add_argument('--landmarks_dir', type=str, default='landmarks',
                       help='Directory containing landmark data')
    parser.add_argument('--config_path', type=str, default='models/landmark_based/model_configs.yaml',
                       help='Path to model configuration file')
    
    args = parser.parse_args()
    
    # Check if directories exist
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory not found: {args.model_dir}")
        return
    
    if not os.path.exists(args.landmarks_dir):
        print(f"Error: Landmarks directory not found: {args.landmarks_dir}")
        return
    
    if not os.path.exists(args.config_path):
        print(f"Error: Config file not found: {args.config_path}")
        return
    
    # Run evaluation
    evaluator = LandmarkModelEvaluator(args.model_dir, args.landmarks_dir, args.config_path)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()