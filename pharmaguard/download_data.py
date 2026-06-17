import os
import cv2
import json
import numpy as np
from utils import Config, get_logger

logger = get_logger(__name__)

def generate_blister_pack_image(filepath, has_defect=False):
    """Generates a synthetic blister pack image. If has_defect is True, simulates a missing pill."""
    # Create background (silver foil)
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200
    noise = np.random.randint(-20, 20, (480, 640, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    rows, cols = 2, 5
    pocket_w, pocket_h = 80, 100
    start_x, start_y = 100, 100
    
    # Pick a random pocket for the defect
    defect_r = np.random.randint(0, rows) if has_defect else -1
    defect_c = np.random.randint(0, cols) if has_defect else -1
    
    for r in range(rows):
        for c in range(cols):
            cx = start_x + c * (pocket_w + 20)
            cy = start_y + r * (pocket_h + 50)
            
            # Draw pocket indent
            cv2.ellipse(img, (cx, cy), (40, 50), 0, 0, 360, (150, 150, 150), 2)
            
            if r == defect_r and c == defect_c:
                # Empty pocket
                cv2.ellipse(img, (cx, cy), (30, 40), 0, 0, 360, (100, 100, 100), -1)
            else:
                # Normal pill
                cv2.ellipse(img, (cx, cy), (30, 40), 0, 0, 360, (240, 250, 250), -1)
                cv2.ellipse(img, (cx+2, cy+2), (30, 40), 0, 0, 360, (180, 180, 180), 2)
                
    cv2.imwrite(filepath, img)

def create_eval_dataset():
    eval_dir = os.path.join(Config.DATA_DIR, "eval_set")
    os.makedirs(eval_dir, exist_ok=True)
    
    labels = {}
    
    for i in range(10):
        filename = f"blister_{i:02d}.jpg"
        filepath = os.path.join(eval_dir, filename)
        
        # 50% chance of defect
        has_defect = np.random.choice([True, False])
        generate_blister_pack_image(filepath, has_defect)
        
        labels[filename] = {
            "defect_detected": has_defect,
            "severity": "high" if has_defect else "none"
        }
        
    labels_path = os.path.join(eval_dir, "labels.json")
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=4)
        
    logger.info(f"Generated 10 synthetic evaluation images and labels.json at {eval_dir}")

if __name__ == "__main__":
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    # Generate the standard single test image for quick demos
    generate_blister_pack_image(os.path.join(Config.DATA_DIR, "synthetic_blister_defect.jpg"), has_defect=True)
    # Generate the full evaluation dataset
    create_eval_dataset()
