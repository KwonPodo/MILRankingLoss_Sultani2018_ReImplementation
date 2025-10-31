# evaluate.py
import argparse
import yaml
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.anomaly_detector import AnomalyDetector
from utils.dataset import C3DFeatureDataset


def load_temporal_annotations(annotation_file):
    """
    Load temporal annotations for test videos.
    
    Returns:
        dict: {video_name: [(start_frame, end_frame), ...]}
    """
    annotations = {}
    
    with open(annotation_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            
            video_name = parts[0].replace('.mp4', '')  # Remove extension
            start1, end1 = int(parts[2]), int(parts[3])
            start2, end2 = int(parts[4]), int(parts[5])
            
            # Store anomaly segments
            segments = []
            if start1 != -1 and end1 != -1:
                segments.append((start1, end1))
            if start2 != -1 and end2 != -1:
                segments.append((start2, end2))
            
            annotations[video_name] = segments
    
    return annotations


def get_frame_level_labels(video_name, annotations, num_segments=32, fps=30):
    """
    Create frame-level binary labels for a video.
    
    Args:
        video_name: name of the video
        annotations: temporal annotations dict
        num_segments: number of segments (32)
        fps: frames per second (30)
    
    Returns:
        np.array: binary labels for each segment (0 or 1)
    """
    labels = np.zeros(num_segments, dtype=np.int32)
    
    # Extract base name (remove class prefix if exists)
    base_name = video_name.split('/')[-1]
    
    if base_name not in annotations:
        # Normal video or not in annotation file
        return labels
    
    # Get anomaly segments
    anomaly_segments = annotations[base_name]
    
    # For each segment, check if it overlaps with any anomaly
    # Assume each segment represents equal portion of the video
    # We don't know exact video length, so we use annotation frames as proxy
    
    if not anomaly_segments:
        return labels
    
    # Find max frame from annotations
    max_frame = max(end for _, end in anomaly_segments)
    
    # Calculate frames per segment
    frames_per_segment = max_frame / num_segments
    
    for seg_idx in range(num_segments):
        seg_start = seg_idx * frames_per_segment
        seg_end = (seg_idx + 1) * frames_per_segment
        
        # Check overlap with any anomaly segment
        for anomaly_start, anomaly_end in anomaly_segments:
            # Check if segment overlaps with anomaly
            if not (seg_end < anomaly_start or seg_start > anomaly_end):
                labels[seg_idx] = 1
                break
    
    return labels


def evaluate_model(model, dataset, annotations, device):
    """
    Evaluate model on test set.
    
    Returns:
        tuple: (all_labels, all_scores)
    """
    model.eval()
    
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Evaluating"):
            sample = dataset[idx]
            features = sample['features'].unsqueeze(0).to(device)  # (1, 32, 4096)
            video_name = sample['video_name']
            
            # Get predictions
            scores = model(features).squeeze(0).cpu().numpy()  # (32,)
            
            # Get ground truth labels
            labels = get_frame_level_labels(video_name, annotations)
            
            all_labels.extend(labels)
            all_scores.extend(scores)
    
    return np.array(all_labels), np.array(all_scores)


def plot_roc_curve(labels, scores, save_path):
    """Plot and save ROC curve"""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"ROC curve saved to {save_path}")
    
    return roc_auc, fpr, tpr, thresholds


def save_results(labels, scores, save_path):
    """Save evaluation results"""
    results = {
        'labels': labels.tolist(),
        'scores': scores.tolist()
    }
    
    import json
    with open(save_path, 'w') as f:
        json.dump(results, f)
    
    print(f"Results saved to {save_path}")


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded config from {args.config}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = AnomalyDetector(
        input_dim=config['model']['input_dim'],
        dropout=config['model']['dropout']
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Loaded model from {args.checkpoint}")
    print(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
    
    # Load test dataset
    test_dataset = C3DFeatureDataset(
        annotation_path=config['data']['test_annotation_path'],
        features_root=config['data']['feature_path']
    )
    
    print(f"Test dataset: {len(test_dataset)} videos")
    
    # Load temporal annotations
    annotations = load_temporal_annotations(args.temporal_annotation)
    print(f"Loaded temporal annotations for {len(annotations)} videos")
    
    # Evaluate
    print("\nEvaluating model...")
    labels, scores = evaluate_model(model, test_dataset, annotations, device)
    
    print(f"\nTotal segments evaluated: {len(labels)}")
    print(f"Anomaly segments: {labels.sum()} ({labels.sum()/len(labels)*100:.1f}%)")
    print(f"Normal segments: {len(labels) - labels.sum()} ({(len(labels)-labels.sum())/len(labels)*100:.1f}%)")
    
    # Calculate ROC-AUC
    print("\nCalculating ROC curve...")
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    roc_auc, fpr, tpr, thresholds = plot_roc_curve(
        labels, scores,
        save_path=results_dir / 'roc_curve.png'
    )
    
    print(f"\n{'='*60}")
    print(f"AUC: {roc_auc:.4f}")
    print(f"{'='*60}")
    
    # Save results
    save_results(labels, scores, results_dir / 'evaluation_results.json')
    
    # Find optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"\nOptimal threshold: {optimal_threshold:.4f}")
    print(f"  TPR: {tpr[optimal_idx]:.4f}")
    print(f"  FPR: {fpr[optimal_idx]:.4f}")
    
    # Save summary
    summary_path = results_dir / 'evaluation_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Evaluation Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Model: {args.checkpoint}\n")
        f.write(f"Test videos: {len(test_dataset)}\n")
        f.write(f"Total segments: {len(labels)}\n")
        f.write(f"Anomaly segments: {labels.sum()} ({labels.sum()/len(labels)*100:.1f}%)\n")
        f.write(f"\nResults:\n")
        f.write(f"  AUC: {roc_auc:.4f}\n")
        f.write(f"  Optimal threshold: {optimal_threshold:.4f}\n")
        f.write(f"  TPR at optimal: {tpr[optimal_idx]:.4f}\n")
        f.write(f"  FPR at optimal: {fpr[optimal_idx]:.4f}\n")
    
    print(f"\nSummary saved to {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate anomaly detection model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--temporal-annotation',
        type=str,
        default='data/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt',
        help='Path to temporal annotation file'
    )
    
    args = parser.parse_args()
    main(args)