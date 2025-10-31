import logging
import wandb
from pathlib import Path

class Logger:
    def __init__(self, config, use_wandb=True):
        self.use_wandb = use_wandb
        
        # Console logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # W&B
        if use_wandb:
            wandb.init(
                project=config['wandb']['project'],
                entity=config['wandb'].get('entity'),
                config=config,
                tags=config['wandb'].get('tags', [])
            )
    
    def log(self, metrics, step=None):
        """Console + W&B Logging"""
        self.logger.info(f"Step {step}: {metrics}")
        if self.use_wandb:
            wandb.log(metrics, step=step)
    
    def log_model(self, model_path):
        """Save model to W&B"""
        if self.use_wandb:
            wandb.save(str(model_path))