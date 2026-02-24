"""
Checkpoint Manager
=====================
Manages model checkpoints for saving and resuming training.
Supports Google Drive integration for Colab persistence.
"""

import os
import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

import torch
from peft import PeftModel

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints with Google Drive support.
    
    Checkpoint Structure:
    ─────────────────────
    checkpoint_epoch_N/
    ├── adapter_model.safetensors  # LoRA weights only
    ├── adapter_config.json        # LoRA configuration
    ├── optimizer.pt               # Optimizer state
    ├── training_state.json        # Epoch, metrics, etc.
    └── metrics_history.json       # Full metrics history
    
    Storage Considerations:
    ───────────────────────
    - Only LoRA adapter weights are saved (~7-15MB each)
    - Base model weights are NOT saved (use model name to reload)
    - Google Drive sync ensures persistence across Colab sessions
    """
    
    def __init__(self, config):
        self.config = config
        self.save_dir = config.save_dir
        self.max_checkpoints = config.max_checkpoints
        self.save_to_drive = config.save_to_drive
        
        # Create checkpoint directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Google Drive mount path (if in Colab)
        self.drive_path = None
        if self.save_to_drive:
            self._setup_drive_path()
        
        self.checkpoints = []
        
        logger.info(f"Checkpoint manager: save_dir={self.save_dir}, "
                     f"max={self.max_checkpoints}, drive={self.save_to_drive}")
    
    def save(
        self,
        model,
        optimizer,
        epoch: int,
        metrics: Optional[Dict] = None,
    ):
        """
        Save a checkpoint.
        
        Saves:
        1. LoRA adapter weights (very small, ~7-15MB)
        2. Optimizer state
        3. Training metadata
        
        Args:
            model: Model with LoRA adapters (PeftModel)
            optimizer: Optimizer with state
            epoch: Current epoch number
            metrics: Optional metrics dictionary
        """
        checkpoint_name = f"checkpoint_epoch_{epoch}"
        checkpoint_path = os.path.join(self.save_dir, checkpoint_name)
        
        logger.info(f"Saving checkpoint: {checkpoint_path}")
        
        try:
            # ── Save LoRA Adapter ──────────────────────────────
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(checkpoint_path)
                logger.info("  LoRA adapter saved")
            
            # ── Save Optimizer State ───────────────────────────
            optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
            torch.save(optimizer.state_dict(), optimizer_path)
            logger.info("  Optimizer state saved")
            
            # ── Save Training State ───────────────────────────
            state = {
                "epoch": epoch,
                "timestamp": datetime.now().isoformat(),
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            }
            if metrics:
                state["metrics"] = metrics
            
            state_path = os.path.join(checkpoint_path, "training_state.json")
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            # ── Track Checkpoint ───────────────────────────────
            self.checkpoints.append(checkpoint_path)
            
            # ── Prune Old Checkpoints ──────────────────────────
            self._prune_old_checkpoints()
            
            # ── Copy to Google Drive ───────────────────────────
            if self.drive_path:
                self._sync_to_drive(checkpoint_path, checkpoint_name)
            
            logger.info(f"  Checkpoint saved successfully: {checkpoint_name}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def load(
        self,
        checkpoint_path: str,
        model,
        optimizer=None,
    ) -> int:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            model: Model to load weights into
            optimizer: Optional optimizer to restore state
            
        Returns:
            Epoch number from the checkpoint
        """
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        try:
            # ── Load LoRA Adapter ──────────────────────────────
            if hasattr(model, 'load_adapter'):
                model.load_adapter(checkpoint_path, adapter_name="default")
                logger.info("  LoRA adapter loaded")
            
            # ── Load Optimizer State ───────────────────────────
            if optimizer:
                optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
                if os.path.exists(optimizer_path):
                    optimizer.load_state_dict(
                        torch.load(optimizer_path, map_location="cpu")
                    )
                    logger.info("  Optimizer state loaded")
            
            # ── Load Training State ────────────────────────────
            state_path = os.path.join(checkpoint_path, "training_state.json")
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                epoch = state.get("epoch", 0)
                logger.info(f"  Resumed from epoch {epoch}")
                return epoch + 1  # Return next epoch to start from
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return 0
    
    def _prune_old_checkpoints(self):
        """Remove oldest checkpoints if we exceed max_checkpoints."""
        while len(self.checkpoints) > self.max_checkpoints:
            oldest = self.checkpoints.pop(0)
            if os.path.exists(oldest):
                shutil.rmtree(oldest)
                logger.info(f"  Pruned old checkpoint: {oldest}")
    
    def _setup_drive_path(self):
        """Setup Google Drive path for checkpoint sync."""
        try:
            # Standard Colab Drive mount point
            drive_base = "/content/drive/MyDrive"
            if os.path.exists(drive_base):
                self.drive_path = os.path.join(
                    drive_base, "SEAL-DSA", "checkpoints"
                )
                os.makedirs(self.drive_path, exist_ok=True)
                logger.info(f"Google Drive sync enabled: {self.drive_path}")
            else:
                logger.info("Google Drive not mounted. Saving locally only.")
                self.drive_path = None
        except Exception as e:
            logger.warning(f"Drive setup failed: {e}")
            self.drive_path = None
    
    def _sync_to_drive(self, src_path: str, name: str):
        """Copy checkpoint to Google Drive."""
        if self.drive_path:
            try:
                dst = os.path.join(self.drive_path, name)
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src_path, dst)
                logger.info(f"  Synced to Drive: {dst}")
            except Exception as e:
                logger.warning(f"Drive sync failed: {e}")
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to the latest checkpoint."""
        if self.checkpoints:
            return self.checkpoints[-1]
        
        # Search save directory
        if os.path.exists(self.save_dir):
            dirs = sorted([
                d for d in os.listdir(self.save_dir)
                if d.startswith("checkpoint_epoch_")
            ])
            if dirs:
                return os.path.join(self.save_dir, dirs[-1])
        
        return None
