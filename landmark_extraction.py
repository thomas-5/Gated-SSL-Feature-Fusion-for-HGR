import os
import yaml
import cv2
import json
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Dict, List, Optional, Tuple


class HandLandmarkExtractor:
    """MediaPipe-based hand landmark extractor for OUHANDS dataset."""
    
    def __init__(self, 
                 static_image_mode: bool = True,
                 max_num_hands: int = 1,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize the MediaPipe hands model.
        
        Args:
            static_image_mode: Whether to treat input as static images
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for landmark tracking
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def extract_landmarks(self, image_path: str) -> Optional[Dict]:
        """
        Extract hand landmarks from a single image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary containing landmarks data or None if no hands detected
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            return None
            
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return None
            
        # Extract landmarks for the first detected hand
        landmarks_data = {
            'image_path': image_path,
            'image_shape': image.shape,
            'landmarks': [],
            'handedness': None
        }
        
        # Get the first hand (since OUHANDS typically has single hand gestures)
        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0] if results.multi_handedness else None
        
        # Extract 21 landmark coordinates
        for landmark in hand_landmarks.landmark:
            landmarks_data['landmarks'].append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z
            })
            
        # Extract handedness information
        if handedness:
            landmarks_data['handedness'] = {
                'label': handedness.classification[0].label,
                'score': handedness.classification[0].score
            }
            
        return landmarks_data
    
    def process_dataset_split(self, 
                            dataset_path: str, 
                            split_name: str, 
                            output_dir: str,
                            file_list: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Process a dataset split and extract landmarks.
        
        Args:
            dataset_path: Path to the dataset root
            split_name: Name of the split ('train', 'validation', or 'test')
            output_dir: Output directory for landmarks
            file_list: Optional list of specific files to process (for validation split)
            
        Returns:
            Dictionary with processing statistics
        """
        stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'no_hands': 0,
            'no_hands_paths': []
        }
        
        # Determine input and output paths
        if split_name in ['train', 'validation']:
            input_dir = os.path.join(dataset_path, 'OUHANDS_train', 'train', 'hand_data', 'colour')
        elif split_name == 'test':
            input_dir = os.path.join(dataset_path, 'OUHANDS_test', 'test', 'hand_data', 'colour')
        else:
            raise ValueError(f"Unknown split: {split_name}")
            
        output_split_dir = os.path.join(output_dir, split_name)
        os.makedirs(output_split_dir, exist_ok=True)
        
        # Get images to process
        if file_list is not None:
            # Use provided file list (for train/validation split)
            image_files = [Path(input_dir) / filename for filename in file_list if (Path(input_dir) / filename).exists()]
        else:
            # Get all PNG images
            image_files = list(Path(input_dir).glob('*.png'))
        
        print(f"\nProcessing {split_name} split: {len(image_files)} images")
        
        # Process each image
        for image_path in tqdm(image_files, desc=f"Extracting landmarks ({split_name})"):
            stats['processed'] += 1
            
            try:
                # Extract landmarks
                landmarks_data = self.extract_landmarks(str(image_path))
                
                # Save landmarks if detected
                if landmarks_data is not None:
                    # Create output filename
                    output_filename = f"{image_path.stem}_landmarks.json"
                    output_path = os.path.join(output_split_dir, output_filename)
                    
                    # Save to JSON
                    with open(output_path, 'w') as f:
                        json.dump(landmarks_data, f, indent=2)
                        
                    stats['successful'] += 1
                else:
                    stats['no_hands_paths'].append(str(image_path))
                    stats['no_hands'] += 1
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                stats['failed'] += 1
                
        return stats
    
    def load_split_files(self, dataset_path: str, split_type: str) -> List[str]:
        """
        Load file list for train/validation splits from OUHANDS predefined splits.
        
        Args:
            dataset_path: Path to the dataset root
            split_type: Either 'training' or 'validation'
            
        Returns:
            List of filenames for the split
        """
        split_file_path = os.path.join(
            dataset_path, 
            'OUHANDS_train', 
            'data_split_for_intermediate_tests', 
            f'{split_type}_files.txt'
        )
        
        if not os.path.exists(split_file_path):
            raise FileNotFoundError(f"Split file not found: {split_file_path}")
            
        with open(split_file_path, 'r') as f:
            files = [line.strip() for line in f if line.strip()]
            
        return files


def main():
    """Main function to run landmark extraction."""
    config_path = 'landmark_configs.yaml'
    try:
        with open(config_path, 'r') as file:
            loaded_config = yaml.safe_load(file) or {}
    except FileNotFoundError:
        loaded_config = {}
        
    parser = argparse.ArgumentParser(description='Extract hand landmarks from OUHANDS dataset')
    parser.add_argument('--dataset_path', type=str, default=loaded_config.get('dataset_path', './dataset'),
                       help='Path to OUHANDS dataset directory')
    parser.add_argument('--output_dir', type=str, default=loaded_config.get('output_dir', './landmarks'),
                       help='Output directory for landmarks')
    parser.add_argument('--detection_confidence', type=float, default=loaded_config.get('detection_confidence', 0.7),
                       help='Minimum detection confidence for MediaPipe')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize extractor
    extractor = HandLandmarkExtractor(
        min_detection_confidence=args.detection_confidence
    )
    
    print("="*60)
    print("OUHANDS Hand Landmark Extraction")
    print("="*60)
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Detection confidence: {args.detection_confidence}")
    print("="*60)
    
    total_stats = {
        'processed': 0,
        'successful': 0,
        'failed': 0,
        'no_hands': 0
    }
    
    # Load predefined train/validation splits
    print("\nLoading predefined train/validation splits...")
    train_files = extractor.load_split_files(args.dataset_path, 'training')
    val_files = extractor.load_split_files(args.dataset_path, 'validation')
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    
    # Process train split
    train_stats = extractor.process_dataset_split(
        args.dataset_path, 'train', args.output_dir, train_files
    )
    
    # Process validation split
    val_stats = extractor.process_dataset_split(
        args.dataset_path, 'validation', args.output_dir, val_files
    )
    
    # Process test split
    test_stats = extractor.process_dataset_split(
        args.dataset_path, 'test', args.output_dir
    )
    
    # Update total stats
    for key in total_stats:
        total_stats[key] = train_stats[key] + val_stats[key] + test_stats[key]
    
    # Print final statistics
    print("\n" + "="*60)
    print("LANDMARK EXTRACTION COMPLETE")
    print("="*60)
    print(f"Total images processed: {total_stats['processed']}")
    print(f"Successful extractions: {total_stats['successful']}")
    print(f"No hands detected: {total_stats['no_hands']}")
    print(f"Failed extractions: {total_stats['failed']}")
    print(f"Success rate: {total_stats['successful']/total_stats['processed']*100:.1f}%")
    print("="*60)
    
    # Save processing report
    report = {
        'dataset_path': args.dataset_path,
        'output_directory': args.output_dir,
        'detection_confidence': args.detection_confidence,
        'train_stats': train_stats,
        'validation_stats': val_stats,
        'test_stats': test_stats,
        'total_stats': total_stats,
        'train_files_count': len(train_files),
        'validation_files_count': len(val_files)
    }
    
    report_path = os.path.join(args.output_dir, 'extraction_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Processing report saved to: {report_path}")


if __name__ == "__main__":
    main()