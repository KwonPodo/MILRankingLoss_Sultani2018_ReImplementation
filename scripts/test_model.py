import torch
from models.anomaly_detector import AnomalyDetector
from models.loss import MILRankingLoss

# Initialize model
model = AnomalyDetector(input_dim=4096, dropout=0.6)
criterion = MILRankingLoss(lambda1=0.00008, lambda2=0.00008)

print("Model architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test forward pass
batch_size = 30
num_segments = 32
feature_dim = 4096

# Dummy data
pos_features = torch.randn(batch_size, num_segments, feature_dim)
neg_features = torch.randn(batch_size, num_segments, feature_dim)

# Forward
model.eval()
with torch.no_grad():
    pos_scores = model(pos_features)
    neg_scores = model(neg_features)

print(f"\nPositive scores shape: {pos_scores.shape}")
print(f"Negative scores shape: {neg_scores.shape}")
print(f"Score range: [{pos_scores.min():.4f}, {pos_scores.max():.4f}]")

# Test loss
loss, loss_dict = criterion(pos_scores, neg_scores)
print(f"\nTotal loss: {loss.item():.4f}")
print(f"  Ranking loss: {loss_dict['ranking_loss']:.4f}")
print(f"  Smoothness loss: {loss_dict['smoothness_loss']:.4f}")
print(f"  Sparsity loss: {loss_dict['sparsity_loss']:.4f}")

# Test training mode
model.train()
pos_scores = model(pos_features)
neg_scores = model(neg_features)
loss, _ = criterion(pos_scores, neg_scores)
print(f"\nTraining mode loss: {loss.item():.4f}")