import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path


class C3DFeatureDataset(Dataset):
    """
    Load pre-extracted C3D features for MIL-based anomaly detection.
    Features are already divided into 32 segments (clips).
    Download C3D features from: https://drive.google.com/drive/folders/1rhOuAdUqyJU4hXIhToUnh5XVvYjQiN50?usp=sharing
    """
    
    def __init__(self, annotation_path, features_root, num_segments=32):
        """
        Args:
            annotation_path: path to train/test annotation file
            features_root: root directory containing feature folders
            num_segments: expected number of segments (should be 32)
        """
        self.features_root = Path(features_root)
        self.num_segments = num_segments
        
        # Parse annotation file
        self.samples = []
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    video_name, label = parts
                    self.samples.append({
                        'video_name': video_name,
                        'label': int(label)
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load C3D features
        feature_path = self.features_root / f"{sample['video_name']}.txt"
        features = self._load_features(feature_path)
        
        return {
            'features': torch.FloatTensor(features),  # (32, 4096)
            'label': sample['label'],
            'video_name': sample['video_name']
        }
    
    def _load_features(self, feature_path):
        """
        Load features from .txt file.
        Expected format: 32 lines, each with 4096 space-separated floats.
        """
        features = np.loadtxt(feature_path, dtype=np.float32)
        
        # Sanity check
        assert features.shape == (self.num_segments, 4096), \
            f"Expected shape (32, 4096), got {features.shape} for {feature_path}"
        
        return features


def collate_fn(batch):
    """
    Custom collate function for MIL.
    Separate positive and negative bags for ranking loss.
    """
    pos_features, neg_features = [], []
    pos_labels, neg_labels = [], []
    pos_names, neg_names = [], []
    
    for item in batch:
        if item['label'] == 1:
            pos_features.append(item['features'])
            pos_labels.append(item['label'])
            pos_names.append(item['video_name'])
        else:
            neg_features.append(item['features'])
            neg_labels.append(item['label'])
            neg_names.append(item['video_name'])
    
    # Stack to tensors
    result = {}
    
    if pos_features:
        result['pos_features'] = torch.stack(pos_features)  # (num_pos, 32, 4096)
        result['pos_labels'] = torch.tensor(pos_labels)
        result['pos_names'] = pos_names
    else:
        result['pos_features'] = None
        result['pos_labels'] = None
        result['pos_names'] = []
    
    if neg_features:
        result['neg_features'] = torch.stack(neg_features)  # (num_neg, 32, 4096)
        result['neg_labels'] = torch.tensor(neg_labels)
        result['neg_names'] = neg_names
    else:
        result['neg_features'] = None
        result['neg_labels'] = None
        result['neg_names'] = []
    
    return result