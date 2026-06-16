import time
import logging
from functools import wraps
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_logger(name):
    return logging.getLogger(name)

def time_it(func):
    """Decorator to measure execution time of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger = get_logger(func.__module__)
        logger.info(f"Execution time for {func.__name__}: {end_time - start_time:.4f} seconds")
        return result
    return wrapper

class Config:
    # Model Configurations
    YOLO_MODEL_PATH = "yolo11n.pt"  # Lightweight YOLO
    
    # VLM configuration
    VLM_MODEL_ID = "microsoft/Florence-2-large"
    
    # LLM configuration
    LLM_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct" # Using a smaller model by default to avoid huge VRAM usage, can swap to 7B.
    
    # Hardware
    DEVICE = "cuda" # ROCm uses the cuda device identifier in PyTorch
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    DEMO_VIDEO_DIR = os.path.join(BASE_DIR, "demo_video_assets")
    
    # Detection Settings
    CONFIDENCE_THRESHOLD = 0.25
