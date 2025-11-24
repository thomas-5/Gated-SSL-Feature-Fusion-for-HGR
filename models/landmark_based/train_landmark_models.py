import os
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import joblib
from datetime import datetime

# Scikit-learn imports
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve
)
import warnings
warnings.filterwarnings('ignore')


class HandLandmarkDataLoader:
    """Load and preprocess hand landmark data from JSON files."""
    
    def __init__(self, landmarks_dir: str, config: Dict):
        self.landmarks_dir = landmarks_dir
        self.config = config
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.pca = None
        
    def load_landmarks_from_json(self, json_path: str) -> Optional[np.ndarray]:
        """Load landmarks from a single JSON file."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            if 'landmarks' not in data or not data['landmarks']:
                return None
                
            # Extract x, y, z coordinates
            landmarks = []
            for landmark in data['landmarks']:
                landmarks.extend([landmark['x'], landmark['y'], landmark['z']])
                
            return np.array(landmarks)
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            return None
    
    def extract_gesture_class(self, filename: str) -> str:
        """Extract gesture class from filename (e.g., 'A-ima-0001_landmarks.json' -> 'A')."""
        return filename.split('-')[0]
    
    def engineer_features(self, landmarks: np.ndarray) -> np.ndarray:
        """Engineer additional features from raw landmarks."""
        features = [landmarks]  # Start with raw coordinates
        
        # Reshape landmarks to (21, 3) for easier manipulation
        landmarks_reshaped = landmarks.reshape(21, 3)
        
        if self.config['feature_engineering']['normalize_coordinates']:
            # Normalize relative to wrist (landmark 0)
            wrist = landmarks_reshaped[0]
            normalized = landmarks_reshaped - wrist
            features.append(normalized.flatten())
        
        if self.config['feature_engineering']['use_distances']:
            # Compute distances between important landmark pairs
            distances = []
            important_pairs = [
                (0, 4), (0, 8), (0, 12), (0, 16), (0, 20),  # Wrist to fingertips
                (4, 8), (8, 12), (12, 16), (16, 20),        # Between fingertips
                (1, 5), (5, 9), (9, 13), (13, 17)          # Between finger bases
            ]
            
            for i, j in important_pairs:
                dist = np.linalg.norm(landmarks_reshaped[i] - landmarks_reshaped[j])
                distances.append(dist)
            
            features.append(np.array(distances))
        
        if self.config['feature_engineering']['use_angles']:
            # Compute angles between landmark triplets
            angles = []
            angle_triplets = [
                (0, 1, 2), (1, 2, 3), (2, 3, 4),  # Thumb angles
                (0, 5, 6), (5, 6, 7), (6, 7, 8),  # Index finger angles
                (0, 9, 10), (9, 10, 11), (10, 11, 12),  # Middle finger angles
                (0, 13, 14), (13, 14, 15), (14, 15, 16),  # Ring finger angles
                (0, 17, 18), (17, 18, 19), (18, 19, 20)   # Pinky angles
            ]
            
            for i, j, k in angle_triplets:
                v1 = landmarks_reshaped[j] - landmarks_reshaped[i]
                v2 = landmarks_reshaped[k] - landmarks_reshaped[j]
                
                # Compute angle between vectors
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                angles.append(angle)
            
            features.append(np.array(angles))
        
        # Concatenate all features
        return np.concatenate(features)
    
    def load_split_data(self, split_name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load data for a specific split (train, validation, test)."""
        split_dir = Path(self.landmarks_dir) / split_name
        
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")
        
        X, y, filenames = [], [], []
        
        # Load all landmark files
        landmark_files = list(split_dir.glob("*_landmarks.json"))
        print(f"Loading {len(landmark_files)} files from {split_name} split...")
        
        for file_path in landmark_files:
            landmarks = self.load_landmarks_from_json(str(file_path))
            if landmarks is not None:
                # Engineer features
                features = self.engineer_features(landmarks)
                X.append(features)
                
                # Extract label
                gesture_class = self.extract_gesture_class(file_path.name)
                y.append(gesture_class)
                filenames.append(file_path.name)
        
        print(f"Successfully loaded {len(X)} samples from {split_name}")
        return np.array(X), np.array(y), filenames
    
    def preprocess_features(self, X_train: np.ndarray, X_val: np.ndarray, 
                          config: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Apply preprocessing (standardization, PCA) to features."""
        
        if config['preprocessing']['standardize']:
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
        
        if config['preprocessing']['pca']['enabled']:
            n_components = config['preprocessing']['pca']['n_components']
            self.pca = PCA(n_components=n_components, random_state=42)
            X_train = self.pca.fit_transform(X_train)
            X_val = self.pca.transform(X_val)
            
            print(f"PCA reduced features from {X_train.shape[1]} to {self.pca.n_components_}")
            print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        return X_train, X_val


class LandmarkModelTrainer:
    """Train and evaluate landmark-based models."""
    
    def __init__(self, config_path: str, landmarks_dir: str, output_dir: str):
        self.config_path = config_path
        self.landmarks_dir = landmarks_dir
        self.output_dir = output_dir
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize data loader
        self.data_loader = HandLandmarkDataLoader(landmarks_dir, self.config['global_settings'])
        
        # Store results
        self.results = {}
        self.models = {}
    
    def create_model(self, model_name: str) -> Any:
        """Create a model based on configuration."""
        model_config = self.config[model_name]
        params = model_config['parameters'].copy()
        
        if model_name == 'svm':
            return SVC(**params)
        elif model_name == 'random_forest':
            return RandomForestClassifier(**params)
        else:
            raise ValueError(f"Unknown model type: {model_name}")
    
    def perform_grid_search(self, model: Any, model_name: str, X_train: np.ndarray, 
                          y_train: np.ndarray) -> Any:
        """Perform grid search for hyperparameter tuning."""
        model_config = self.config[model_name]
        
        if not model_config['training']['grid_search']['enabled']:
            return model
        
        param_grid = model_config['training']['grid_search']['param_grid'].copy()
        
        # Handle special parameter conversions for MLP
        if model_name == 'mlp' and 'hidden_layer_sizes' in param_grid:
            # Convert hidden_layer_sizes lists to tuples for MLPClassifier
            param_grid['hidden_layer_sizes'] = [
                tuple(sizes) if isinstance(sizes, list) else sizes 
                for sizes in param_grid['hidden_layer_sizes']
            ]
        
        cv_folds = model_config['training']['cv_folds']
        
        print(f"Performing grid search for {model_name}...")
        
        # Create stratified k-fold
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=skf, scoring='accuracy',
            n_jobs=-1, verbose=1, return_train_score=True
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str) -> Dict:
        """Evaluate model and compute metrics."""
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Classification report
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
        
        # AUC score for multi-class (if probabilities available)
        if y_pred_proba is not None:
            try:
                auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                results['auc_score'] = auc_score
            except Exception as e:
                print(f"Could not compute AUC score: {e}")
        
        return results
    
    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str, classes: List[str]):
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Save plot
        plot_path = os.path.join(self.output_dir, f'{model_name}_confusion_matrix.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved: {plot_path}")
    
    def train_and_evaluate_model(self, model_name: str, X_train: np.ndarray, 
                               y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """Train and evaluate a single model."""
        
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()} Model")
        print(f"{'='*60}")
        
        # Preprocess features
        model_config = self.config[model_name]
        X_train_processed, X_val_processed = self.data_loader.preprocess_features(
            X_train.copy(), X_val.copy(), model_config
        )
        
        print(f"Feature shape after preprocessing: {X_train_processed.shape}")
        
        # Create model
        model = self.create_model(model_name)
        
        # Perform grid search if enabled
        model = self.perform_grid_search(model, model_name, X_train_processed, y_train)
        
        # Train final model
        print(f"Training final {model_name} model...")
        model.fit(X_train_processed, y_train)
        
        # Evaluate on validation set
        results = self.evaluate_model(model, X_val_processed, y_val, model_name)
        
        # Store results and model
        self.results[model_name] = results
        self.models[model_name] = {
            'model': model,
            'scaler': self.data_loader.scaler if model_config['preprocessing']['standardize'] else None,
            'pca': self.data_loader.pca if model_config['preprocessing']['pca']['enabled'] else None
        }
        
        # Print results
        print(f"\n{model_name.upper()} Results:")
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1-Score:  {results['f1_score']:.4f}")
        if 'auc_score' in results:
            print(f"AUC Score: {results['auc_score']:.4f}")
        
        # Generate plots
        if self.config['global_settings']['output']['generate_plots']:
            classes = self.config['dataset']['classes']
            self.plot_confusion_matrix(np.array(results['confusion_matrix']), model_name, classes)
        
        return results
    
    def save_results(self):
        """Save all results and models."""
        
        # Save results as JSON
        results_path = os.path.join(self.output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved: {results_path}")
        
        # Save models
        if self.config['global_settings']['output']['save_models']:
            for model_name, model_data in self.models.items():
                model_path = os.path.join(self.output_dir, f'{model_name}_model.joblib')
                joblib.dump(model_data, model_path)
                print(f"Model saved: {model_path}")
        
        # Create summary report
        self.create_summary_report()
    
    def create_summary_report(self):
        """Create a summary comparison report."""
        
        summary_data = []
        for model_name, results in self.results.items():
            summary_data.append({
                'Model': model_name.upper(),
                'Accuracy': f"{results['accuracy']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}",
                'F1-Score': f"{results['f1_score']:.4f}",
                'AUC': f"{results.get('auc_score', 'N/A'):.4f}" if isinstance(results.get('auc_score'), float) else 'N/A'
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save as CSV
        summary_path = os.path.join(self.output_dir, 'model_comparison_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        # Print summary
        print(f"\n{'='*80}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(summary_df.to_string(index=False))
        print(f"{'='*80}")
        
        print(f"\nSummary saved: {summary_path}")
    
    def run_training(self):
        """Run the complete training and evaluation pipeline."""
        
        print("="*80)
        print("LANDMARK-BASED HAND GESTURE RECOGNITION TRAINING")
        print("="*80)
        print(f"Landmarks directory: {self.landmarks_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Config file: {self.config_path}")
        
        # Load data
        print("\nLoading training data...")
        X_train, y_train, train_files = self.data_loader.load_split_data('train')
        
        print("Loading validation data...")
        X_val, y_val, val_files = self.data_loader.load_split_data('validation')
        
        # Encode labels
        y_train_encoded = self.data_loader.label_encoder.fit_transform(y_train)
        y_val_encoded = self.data_loader.label_encoder.transform(y_val)
        
        print(f"\nDataset Statistics:")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Feature dimension: {X_train.shape[1]}")
        print(f"Classes: {list(self.data_loader.label_encoder.classes_)}")
        
        # Train models
        models_to_train = ['svm', 'random_forest']
        
        for model_name in models_to_train:
            try:
                self.train_and_evaluate_model(
                    model_name, X_train, y_train_encoded, X_val, y_val_encoded
                )
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
        
        # Save results
        self.save_results()
        
        print(f"\n{'='*80}")
        print("TRAINING COMPLETED!")
        print(f"{'='*80}")


def main():
    """Main function to run landmark-based model training."""
    
    # Paths
    config_path = "models/landmark_based/model_configs.yaml"
    landmarks_dir = "landmarks"
    output_dir = f"models/landmark_based/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Check if landmarks exist
    if not os.path.exists(landmarks_dir):
        print(f"Error: Landmarks directory not found: {landmarks_dir}")
        print("Please run landmark extraction first: python landmark_extraction.py")
        return
    
    # Check if config exists
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        return
    
    # Create trainer and run
    trainer = LandmarkModelTrainer(config_path, landmarks_dir, output_dir)
    trainer.run_training()

if __name__ == "__main__":
    main()