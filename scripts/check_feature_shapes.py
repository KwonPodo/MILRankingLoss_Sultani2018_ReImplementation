from pathlib import Path
import numpy as np

features_root = Path('data/features')

shapes = []
for cls_folder in ['Fighting', 'Robbery', 'Stealing', 'Training_Normal_Videos_Anomaly']:
    cls_path = features_root / cls_folder
    if not cls_path.exists():
        continue
    
    for txt_file in cls_path.glob('*.txt'):
        data = np.loadtxt(txt_file)
        shapes.append((cls_folder, txt_file.name, data.shape))

# Print statistics
print(f"Total files: {len(shapes)}")
print(f"\nShape distribution:")

unique_shapes = {}
for cls, name, shape in shapes:
    if shape not in unique_shapes:
        unique_shapes[shape] = []
    unique_shapes[shape].append((cls, name))

for shape, files in sorted(unique_shapes.items()):
    print(f"  {shape}: {len(files)} files")
    if len(files) <= 3:
        for cls, name in files:
            print(f"    - {cls}/{name}")