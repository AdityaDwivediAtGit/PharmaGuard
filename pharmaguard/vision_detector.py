import cv2
import numpy as np
import torch
from ultralytics import YOLO
from utils import time_it, get_logger, Config

logger = get_logger(__name__)

class VisionDetector:
    def __init__(self, model_path=Config.YOLO_MODEL_PATH):
        logger.info(f"Loading YOLO model from {model_path}...")
        try:
            device = Config.DEVICE
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA is not available; falling back to CPU for YOLO.")
                device = "cpu"
            self.model = YOLO(model_path, device=device)
            self.model.to(device)
            logger.info(f"YOLO model loaded successfully on {device}.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None

    @time_it
    def process_frame(self, frame):
        """
        Process a single video frame. Detect and track objects.
        Returns the annotated frame, a list of cropped objects, and metadata.
        """
        if self.model is None:
            return frame, [], []
        
        # Run YOLO inference with tracking
        results = self.model.track(frame, persist=True, conf=Config.CONFIDENCE_THRESHOLD, verbose=False)
        
        annotated_frame = frame.copy()
        crops = []
        metadata = []
        
        if results and len(results) > 0:
            result = results[0]
            annotated_frame = result.plot()
            
            if result.boxes:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    track_id = int(box.id[0]) if box.id is not None else -1
                    
                    # Ensure coordinates are within frame bounds
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2]
                        crops.append(crop)
                        
                        metadata.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": conf,
                            "class_id": cls_id,
                            "track_id": track_id
                        })
                        
        return annotated_frame, crops, metadata
