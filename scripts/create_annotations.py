from pathlib import Path

def load_test_videos(test_split_file):
    """Load test video names from Anomaly_Test.txt"""
    test_videos = set()
    with open(test_split_file, 'r') as f:
        for line in f:
            video_path = line.strip()
            if video_path:
                # e.g., Fighting/Fighting003_x264.mp4 → Fighting003_x264
                video_name = Path(video_path).stem
                test_videos.add(video_name)
    return test_videos

def create_annotations(features_path, original_test_split):
    """
    Create train/test annotations based on original UCF-Crime split.
    Train = all videos except those in Anomaly_Test.txt
    Test = videos in Anomaly_Test.txt
    """
    # Load test videos
    test_videos = load_test_videos(original_test_split)
    print(f"Test videos from original split: {len(test_videos)}")
    
    features_path = Path(features_path)
    train_lines = []
    test_lines = []

    anomaly_classes = []
    for cls in features_path.iterdir():
        if cls.is_dir() and cls.name not in ['Testing_Normal_Videos_Anomaly', 'Training_Normal_Videos_Anomaly']:
            anomaly_classes.append(cls.name)
    
    print(anomaly_classes)
    
    # Process anomaly videos
    for cls in anomaly_classes:
        cls_path = features_path / cls
        if not cls_path.exists():
            print(f"Warning: {cls} folder not found")
            continue
        
        train_count = 0
        test_count = 0
        
        for feature_file in sorted(cls_path.glob("*.txt")):
            video_name = feature_file.stem  # e.g., Fighting003_x264
            
            if video_name in test_videos:
                # Test set
                test_lines.append(f"{cls}/{video_name} 1\n")
                test_count += 1
            else:
                # Train set
                train_lines.append(f"{cls}/{video_name} 1\n")
                train_count += 1
        
        print(f"{cls}: {train_count} train, {test_count} test")
    
    # Normal videos
    train_normal = features_path / "Training_Normal_Videos_Anomaly"
    if train_normal.exists():
        normal_count = 0
        for feature_file in sorted(train_normal.glob("*.txt")):
            train_lines.append(f"Training_Normal_Videos_Anomaly/{feature_file.stem} 0\n")
            normal_count += 1
        print(f"Training_Normal: {normal_count} videos")
    
    test_normal = features_path / "Testing_Normal_Videos_Anomaly"
    if test_normal.exists():
        normal_count = 0
        for feature_file in sorted(test_normal.glob("*.txt")):
            test_lines.append(f"Testing_Normal_Videos_Anomaly/{feature_file.stem} 0\n")
            normal_count += 1
        print(f"Testing_Normal: {normal_count} videos")
    
    # Save
    Path("data/annotations").mkdir(parents=True, exist_ok=True)
    
    with open("data/annotations/train_set.txt", "w") as f:
        f.writelines(train_lines)
    with open("data/annotations/test_set.txt", "w") as f:
        f.writelines(test_lines)
    
    # Statistics
    train_pos = sum(1 for line in train_lines if line.endswith('1\n'))
    train_neg = len(train_lines) - train_pos
    test_pos = sum(1 for line in test_lines if line.endswith('1\n'))
    test_neg = len(test_lines) - test_pos
    
    print(f"\n{'='*70}")
    print("FINAL SPLIT")
    print('='*70)
    print(f"Train: {len(train_lines)} videos")
    print(f"  Anomaly: {train_pos}")
    print(f"  Normal:  {train_neg}")
    print(f"  Batches per epoch (30+30): {min(train_pos, train_neg) // 30}")
    
    print(f"\nTest:  {len(test_lines)} videos")
    print(f"  Anomaly: {test_pos}")
    print(f"  Normal:  {test_neg}")
    print('='*70)
    
    # Warnings
    if test_pos == 0:
        print("\n⚠️  WARNING: No anomaly videos in test set!")
    if min(train_pos, train_neg) < 30:
        print(f"\n⚠️  WARNING: Not enough samples for batch_size=60")
        print(f"   Consider reducing batch_size or adding more classes")

if __name__ == "__main__":
    create_annotations(
        features_path="data/features",
        original_test_split="/mnt/hdd1/UCF-Crime/Anomaly_Detection_splits/Anomaly_Test.txt"
    )