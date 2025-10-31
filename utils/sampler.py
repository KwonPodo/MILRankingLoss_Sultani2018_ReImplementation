import random
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):
    """
    Sample equal number of positive and negative samples per batch.
    """
    
    def __init__(self, dataset, batch_size=60):
        """
        Args:
            dataset: C3DFeatureDataset instance
            batch_size: total batch size (must be even)
        """
        assert batch_size % 2 == 0, "Batch size must be even"
        
        self.batch_size = batch_size
        self.samples_per_class = batch_size // 2  # 30 each
        
        # Separate positive and negative indices
        self.pos_indices = []
        self.neg_indices = []
        
        for idx, sample in enumerate(dataset.samples):
            if sample['label'] == 1:
                self.pos_indices.append(idx)
            else:
                self.neg_indices.append(idx)
        
        print(f"Positive samples: {len(self.pos_indices)}")
        print(f"Negative samples: {len(self.neg_indices)}")
        
        # Calculate number of batches
        self.num_batches = min(
            len(self.pos_indices) // self.samples_per_class,
            len(self.neg_indices) // self.samples_per_class
        )
    
    def __iter__(self):
        # Shuffle indices
        pos_shuffled = random.sample(self.pos_indices, len(self.pos_indices))
        neg_shuffled = random.sample(self.neg_indices, len(self.neg_indices))
        
        for i in range(self.num_batches):
            # Get 30 positive and 30 negative
            pos_batch = pos_shuffled[i * self.samples_per_class : (i + 1) * self.samples_per_class]
            neg_batch = neg_shuffled[i * self.samples_per_class : (i + 1) * self.samples_per_class]
            
            # Combine and shuffle
            batch = pos_batch + neg_batch
            random.shuffle(batch)
            
            yield batch
    
    def __len__(self):
        return self.num_batches