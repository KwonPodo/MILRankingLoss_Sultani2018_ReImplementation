from utils.dataset import C3DFeatureDataset, collate_fn
from utils.sampler import BalancedBatchSampler
from torch.utils.data import DataLoader

dataset = C3DFeatureDataset(
    annotation_path="data/annotations/train_subset.txt",
    features_root="data/features"
) 

print(f'Total samples in dataset: {len(dataset)}')

sampler = BalancedBatchSampler(dataset, batch_size=60)


loader = DataLoader(
    dataset,
    batch_sampler=sampler,
    collate_fn=collate_fn
)


batch = next(iter(loader))
print(f'\nFirst batch:')
print(f'\nPositive bags: {batch["pos_features"].shape}')
print(f'\nNegative bags: {batch["neg_features"].shape}')

# Check multiple batches
for i, batch in enumerate(loader):
    pos_size = batch['pos_features'].shape[0] if batch['pos_features'] is not None else 0
    neg_size = batch['neg_features'].shape[0] if batch['neg_features'] is not None else 0
    print(f"Batch {i}: pos={pos_size}, neg={neg_size}")
    
    if i >= 3:  # Check first 4 batches
        break