"""
RDS6 Configuration for VisualSearchLLM
Manages storage paths for experiments, results, logs, and other data.
"""

import os
from pathlib import Path
from datetime import datetime

# RDS6 base path
RDS6_BASE = "/rds-d6/user/mm2833/hpc-work/VisualSearchLLM"

# Directory structure
RDS6_DIRS = {
    "experiments": f"{RDS6_BASE}/experiments",
    "results": f"{RDS6_BASE}/results", 
    "logs": f"{RDS6_BASE}/logs",
    "checkpoints": f"{RDS6_BASE}/checkpoints",
    "images": f"{RDS6_BASE}/images",
    "models": f"{RDS6_BASE}/models"
}

def ensure_rds6_dirs():
    """Create RDS6 directories if they don't exist."""
    for dir_name, dir_path in RDS6_DIRS.items():
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Ensured RDS6 directory: {dir_path}")

def get_experiment_path(experiment_name, subdir="results"):
    """Get path for a specific experiment."""
    if subdir not in RDS6_DIRS:
        raise ValueError(f"Invalid subdir: {subdir}. Must be one of {list(RDS6_DIRS.keys())}")
    
    experiment_path = Path(RDS6_DIRS[subdir]) / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)
    return str(experiment_path)

def get_log_path(experiment_name):
    """Get log file path for an experiment."""
    log_dir = Path(RDS6_DIRS["logs"]) / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(log_dir / f"experiment_{timestamp}.log")

def get_checkpoint_path(experiment_name, checkpoint_name):
    """Get checkpoint file path."""
    checkpoint_dir = Path(RDS6_DIRS["checkpoints"]) / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return str(checkpoint_dir / checkpoint_name)

def get_image_path(experiment_name, image_name):
    """Get image file path."""
    image_dir = Path(RDS6_DIRS["images"]) / experiment_name
    image_dir.mkdir(parents=True, exist_ok=True)
    return str(image_dir / image_name)

def get_model_path(experiment_name, model_name):
    """Get model file path.

    If environment variable VSL_MODELS_DIR is set, write under that directory
    instead of the default RDS6 models path. This enables local artifact storage.
    """
    override_root = os.environ.get("VSL_MODELS_DIR")
    if override_root:
        model_dir = Path(override_root) / experiment_name
    else:
        model_dir = Path(RDS6_DIRS["models"]) / experiment_name
    model_dir.mkdir(parents=True, exist_ok=True)
    return str(model_dir / model_name)

def sync_to_rds6(local_path, experiment_name, subdir="results"):
    """Sync local files to RDS6 storage."""
    import shutil
    
    local_path = Path(local_path)
    rds6_path = Path(get_experiment_path(experiment_name, subdir))
    
    if local_path.is_file():
        # Copy single file
        shutil.copy2(local_path, rds6_path / local_path.name)
        print(f"✓ Copied {local_path} to RDS6: {rds6_path / local_path.name}")
    elif local_path.is_dir():
        # Copy directory contents
        for item in local_path.iterdir():
            if item.is_file():
                shutil.copy2(item, rds6_path / item.name)
            elif item.is_dir():
                shutil.copytree(item, rds6_path / item.name, dirs_exist_ok=True)
        print(f"✓ Synced {local_path} to RDS6: {rds6_path}")

def get_rds6_status():
    """Get status of RDS6 directories and storage usage."""
    status = {}
    for dir_name, dir_path in RDS6_DIRS.items():
        path = Path(dir_path)
        if path.exists():
            try:
                # Get directory size (recursive)
                total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                status[dir_name] = {
                    "path": dir_path,
                    "exists": True,
                    "size_mb": round(total_size / (1024 * 1024), 2),
                    "file_count": len(list(path.rglob('*')))
                }
            except Exception as e:
                status[dir_name] = {
                    "path": dir_path,
                    "exists": True,
                    "error": str(e)
                }
        else:
            status[dir_name] = {
                "path": dir_path,
                "exists": False
            }
    return status

if __name__ == "__main__":
    # Test and initialize RDS6 setup
    print("🧪 Testing RDS6 setup for VisualSearchLLM...")
    ensure_rds6_dirs()
    
    print("\n📊 RDS6 Status:")
    status = get_rds6_status()
    for dir_name, info in status.items():
        if info.get("exists"):
            size = info.get("size_mb", 0)
            count = info.get("file_count", 0)
            print(f"  {dir_name}: {size} MB, {count} files")
        else:
            print(f"  {dir_name}: Not created")
    
    print("\n🎉 RDS6 setup complete!")
