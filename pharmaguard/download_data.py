import os
import cv2
import numpy as np
from utils import Config, get_logger

logger = get_logger(__name__)

def generate_synthetic_blister_pack(filename="synthetic_blister_defect.jpg"):
    """
    Generates a synthetic blister pack image with a simulated 'missing pill' defect
    for testing the pipeline out-of-the-box without requiring dataset downloads.
    """
    filepath = os.path.join(Config.DATA_DIR, filename)
    
    # Create background (silver foil)
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200
    
    # Add some noise to simulate foil texture
    noise = np.random.randint(-20, 20, (480, 640, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Draw 2x5 grid of blister pockets
    rows, cols = 2, 5
    pocket_w, pocket_h = 80, 100
    start_x, start_y = 100, 100
    
    for r in range(rows):
        for c in range(cols):
            cx = start_x + c * (pocket_w + 20)
            cy = start_y + r * (pocket_h + 50)
            
            # Draw pocket indent
            cv2.ellipse(img, (cx, cy), (40, 50), 0, 0, 360, (150, 150, 150), 2)
            
            # Simulate a defect: missing pill at row 1, col 2
            if r == 1 and c == 2:
                # Empty pocket (just draw the shadow)
                cv2.ellipse(img, (cx, cy), (30, 40), 0, 0, 360, (100, 100, 100), -1)
                # cv2.putText(img, "Empty", (cx - 20, cy - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                # Normal pill (white/yellowish)
                cv2.ellipse(img, (cx, cy), (30, 40), 0, 0, 360, (240, 250, 250), -1)
                # Shadow to make it look 3D
                cv2.ellipse(img, (cx+2, cy+2), (30, 40), 0, 0, 360, (180, 180, 180), 2)
                
    cv2.imwrite(filepath, img)
    logger.info(f"Generated synthetic blister pack image at {filepath}")

if __name__ == "__main__":
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    generate_synthetic_blister_pack()
    
    print("\n--- Note ---")
    print("To download real public datasets (e.g., from Roboflow):")
    print("1. pip install roboflow")
    print("2. Use your Roboflow API key to download a pharmaceutical blister pack dataset.")
