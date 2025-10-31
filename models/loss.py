import torch
import torch.nn as nn


class MILRankingLoss(nn.Module):
    """
    Multiple Instance Learning Ranking Loss with sparsity and smoothness constraints.
    
    Loss formula from paper:
    loss = hinge_loss + λ1 * smoothness + λ2 * sparsity
    
    where:
    - hinge_loss = max(0, 1 - max(pos_scores) + max(neg_scores))
    - smoothness = sum of squared differences between adjacent segments
    - sparsity = sum of all positive bag scores
    """
    
    def __init__(self, lambda1=0.00008, lambda2=0.00008):
        """
        Args:
            lambda1: weight for temporal smoothness constraint
            lambda2: weight for sparsity constraint
        """
        super(MILRankingLoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
    
    def forward(self, pos_scores, neg_scores):
        """
        Args:
            pos_scores: (batch_pos, num_segments) - scores for positive bags
            neg_scores: (batch_neg, num_segments) - scores for negative bags
        
        Returns:
            loss: scalar tensor
        """
        # MIL ranking loss: max score of positive bag should be higher than negative
        pos_max = torch.max(pos_scores, dim=1)[0]  # (batch_pos,)
        neg_max = torch.max(neg_scores, dim=1)[0]  # (batch_neg,)
        
        # Hinge loss
        ranking_loss = torch.clamp(
            1.0 - pos_max.mean() + neg_max.mean(),
            min=0
        )
        
        # Temporal smoothness: minimize difference between adjacent segments
        smoothness_loss = 0
        if pos_scores.size(1) > 1:  # if more than 1 segment
            temporal_diff = pos_scores[:, 1:] - pos_scores[:, :-1]  # (batch, 31)
            smoothness_loss = torch.sum(temporal_diff ** 2)
        
        # Sparsity: minimize sum of all scores (encourage sparse anomalies)
        sparsity_loss = torch.sum(pos_scores)
        
        # Total loss
        total_loss = ranking_loss + self.lambda1 * smoothness_loss + self.lambda2 * sparsity_loss
        
        return total_loss, {
            'ranking_loss': ranking_loss.item(),
            'smoothness_loss': smoothness_loss.item() if isinstance(smoothness_loss, torch.Tensor) else 0.0,
            'sparsity_loss': sparsity_loss.item()
        }