"""
Google Colab Utilities
========================
Helper functions for Google Colab-specific operations:
- Environment detection
- Drive mounting
- GPU monitoring
- Session management
"""

import os
import logging
import subprocess

logger = logging.getLogger(__name__)


def is_colab() -> bool:
    """Check if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def setup_colab_environment():
    """
    Setup Colab-specific configurations.
    
    1. Mount Google Drive (for checkpoint persistence)
    2. Install additional packages if needed
    3. Set memory optimization flags
    """
    if not is_colab():
        logger.info("Not in Colab, skipping setup")
        return
    
    # Mount Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        logger.info("Google Drive mounted successfully")
    except Exception as e:
        logger.warning(f"Drive mount failed: {e}")
    
    # Set PyTorch memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    logger.info("Colab environment configured")


def get_gpu_info() -> dict:
    """Get detailed GPU information."""
    info = {
        "available": False,
        "name": "N/A",
        "memory_total": "N/A",
        "memory_used": "N/A",
        "memory_free": "N/A",
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            info["available"] = True
            info["name"] = torch.cuda.get_device_name(0)
            
            total = torch.cuda.get_device_properties(0).total_mem / 1024**3
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            
            info["memory_total"] = f"{total:.1f} GB"
            info["memory_used"] = f"{allocated:.1f} GB"
            info["memory_free"] = f"{total - allocated:.1f} GB"
    except Exception as e:
        logger.debug(f"GPU info error: {e}")
    
    return info


def install_dependencies():
    """Install required packages in Colab."""
    packages = [
        "transformers>=4.36.0",
        "peft>=0.7.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "datasets>=2.16.0",
        "evaluate>=0.4.0",
        "rouge-score",
        "wandb",
    ]
    
    for pkg in packages:
        try:
            subprocess.run(
                ["pip", "install", "-q", pkg],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to install {pkg}: {e}")
    
    logger.info("Dependencies installed")


def colab_keep_alive():
    """
    Prevent Colab session from disconnecting.
    
    Note: This uses the standard approach of creating
    periodic activity. Be mindful of Colab's usage policies.
    """
    if not is_colab():
        return
    
    try:
        from google.colab import output
        output.eval_js('''
            function keepAlive() {
                document.querySelector("colab-connect-button").click()
            }
            setInterval(keepAlive, 60000)
        ''')
        logger.info("Keep-alive activated")
    except Exception as e:
        logger.warning(f"Keep-alive setup failed: {e}")


def save_to_drive(src_path: str, drive_subpath: str = "SEAL-DSA"):
    """
    Save a file or directory to Google Drive.
    
    Args:
        src_path: Local path to save
        drive_subpath: Subdirectory in Google Drive
    """
    import shutil
    
    drive_base = "/content/drive/MyDrive"
    if not os.path.exists(drive_base):
        logger.warning("Google Drive not mounted")
        return
    
    dst = os.path.join(drive_base, drive_subpath, os.path.basename(src_path))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    
    if os.path.isdir(src_path):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src_path, dst)
    else:
        shutil.copy2(src_path, dst)
    
    logger.info(f"Saved to Drive: {dst}")
