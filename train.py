import argparse
import yaml
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from models.anomaly_detector import AnomalyDetector
from models.loss import MILRankingLoss
from utils.dataset import C3DFeatureDataset, collate_fn
from utils.sampler import BalancedBatchSampler


def load_config(config_path):
    """Load configuration from yaml file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_wandb(config):
    """Initialize Weights & Biases"""
    wandb.init(
        project=config['wandb']['project'],
        config=config
    )


def build_model(config, device):
    """Build model and move to device"""
    model = AnomalyDetector(
        input_dim=config['model']['input_dim'],
        dropout=config['model']['dropout']
    )
    model = model.to(device)
    return model


def build_optimizer(model, config):
    """Build optimizer"""
    optimizer_name = config['training']['optimizer'].lower()
    lr = config['training']['learning_rate']
    weight_decay = config['training']['lambda3']
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train one epoch"""
    model.train()
    
    epoch_loss = 0.0
    epoch_ranking_loss = 0.0
    epoch_smoothness_loss = 0.0
    epoch_sparsity_loss = 0.0
    
    progress_bar = tqdm(loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        pos_features = batch['pos_features']
        neg_features = batch['neg_features']
        
        if pos_features is None or neg_features is None:
            continue
        
        pos_features = pos_features.to(device)
        neg_features = neg_features.to(device)
        
        # Forward
        pos_scores = model(pos_features)
        neg_scores = model(neg_features)
        
        # Loss
        loss, loss_dict = criterion(pos_scores, neg_scores)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        epoch_loss += loss.item()
        epoch_ranking_loss += loss_dict['ranking_loss']
        epoch_smoothness_loss += loss_dict['smoothness_loss']
        epoch_sparsity_loss += loss_dict['sparsity_loss']
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'rank': f"{loss_dict['ranking_loss']:.4f}"
        })
    
    num_batches = len(loader)
    avg_loss = epoch_loss / num_batches
    avg_ranking = epoch_ranking_loss / num_batches
    avg_smoothness = epoch_smoothness_loss / num_batches
    avg_sparsity = epoch_sparsity_loss / num_batches
    
    return {
        'loss': avg_loss,
        'ranking_loss': avg_ranking,
        'smoothness_loss': avg_smoothness,
        'sparsity_loss': avg_sparsity
    }


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, save_path)


def main(args):
    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup W&B
    if not args.no_wandb:
        setup_wandb(config)
        print("Weights & Biases initialized")
    
    # Create checkpoint directory
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Build dataset
    train_dataset = C3DFeatureDataset(
        annotation_path=config['data']['train_annotation_path'],
        features_root=config['data']['feature_path']
    )
    print(f"Train dataset: {len(train_dataset)} videos")
    
    # Build sampler and loader
    sampler = BalancedBatchSampler(
        train_dataset,
        batch_size=config['training']['batch_size']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=4
    )
    print(f"Total batches per epoch: {len(train_loader)}")
    
    # Build model
    model = build_model(config, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Build optimizer
    optimizer = build_optimizer(model, config)
    print(f"Optimizer: {config['training']['optimizer']}")
    
    # Build loss
    criterion = MILRankingLoss(
        lambda1=config['training']['lambda1'],
        lambda2=config['training']['lambda2']
    )
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    best_loss = float('inf')
    
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(1, num_epochs + 1):
        metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Ranking: {metrics['ranking_loss']:.4f}")
        print(f"  Smoothness: {metrics['smoothness_loss']:.4f}")
        print(f"  Sparsity: {metrics['sparsity_loss']:.4f}")
        
        # Log to W&B
        if not args.no_wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss': metrics['loss'],
                'train/ranking_loss': metrics['ranking_loss'],
                'train/smoothness_loss': metrics['smoothness_loss'],
                'train/sparsity_loss': metrics['sparsity_loss']
            })
        
        # Save checkpoint
        if epoch % 10 == 0 or metrics['loss'] < best_loss:
            save_path = checkpoint_dir / f'epoch_{epoch}.pth'
            save_checkpoint(model, optimizer, epoch, metrics['loss'], save_path)
            print(f"  Saved checkpoint: {save_path}")
            
            if metrics['loss'] < best_loss:
                best_loss = metrics['loss']
                best_path = checkpoint_dir / 'best_model.pth'
                save_checkpoint(model, optimizer, epoch, metrics['loss'], best_path)
                print(f"  New best model: {best_path}")
    
    print("\nTraining completed!")
    
    if not args.no_wandb:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train anomaly detection model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )
    
    args = parser.parse_args()
    main(args)