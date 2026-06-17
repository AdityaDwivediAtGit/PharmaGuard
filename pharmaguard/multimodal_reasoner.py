import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import numpy as np
import cv2
from utils import time_it, get_logger, Config

logger = get_logger(__name__)

class MultimodalReasoner:
    def __init__(self, model_id="Salesforce/blip-vqa-base"):
        # Switched to BLIP VQA because Florence-2 remote code is incompatible with the latest 
        # transformers required by Qwen2.5, causing the 'forced_bos_token_id' AttributeError.
        # BLIP is natively supported and runs significantly faster on CPU.
        logger.info(f"Loading VLM model from {model_id}...")
        try:
            self.device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
            self.processor = BlipProcessor.from_pretrained(model_id)
            self.model = BlipForQuestionAnswering.from_pretrained(
                model_id, 
                torch_dtype=torch.float16 if self.device.type != "cpu" else torch.float32,
            ).to(self.device)
            logger.info(f"VLM model {model_id} loaded successfully on {self.device}.")
        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            logger.error(f"Failed to load VLM model: {err_msg}")
            self.model = None
            self.load_error = str(e)

    @time_it
    def analyze_crop(self, crop, task_prompt="Are there any missing, empty, or broken pockets in this blister pack? Answer in detail."):
        """
        Analyze a cropped image of a blister pack/tablet using BLIP VQA.
        """
        if self.model is None:
            return self._fallback_description(crop)
        if crop is None or crop.size == 0:
            return "Invalid crop (empty image)."
        
        # Convert OpenCV BGR image to RGB PIL Image
        if isinstance(crop, np.ndarray):
            rgb_image = crop[:, :, ::-1]
            image = Image.fromarray(rgb_image)
        else:
            image = crop

        inputs = self.processor(image, task_prompt, return_tensors="pt")
        
        # Move inputs to device and handle dtype for pixel_values
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs and self.device.type != "cpu":
             inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=100
        )
        
        generated_text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text

    def _fallback_description(self, crop):
        if crop is None or crop.size == 0:
            return "VLM model not loaded. ERROR: Unknown. Invalid crop."

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mean = gray.mean()
        std = gray.std()
        dark_spots = np.sum(gray < 50)
        bright_spots = np.sum(gray > 220)
        empty_pocket_hint = "likely empty pocket" if mean < 120 and bright_spots < 10 else "no obvious empty pocket detected"
        return (
            f"VLM model not loaded. ERROR: {getattr(self, 'load_error', 'Unknown')}. "
            f"Fallback description: mean intensity={mean:.1f}, contrast={std:.1f}, {empty_pocket_hint}."
        )

    def detect_defects(self, crop):
        """
        Specific prompt for defect detection. We ask for a detailed caption and then
        the LLM will analyze if there are defects based on that description.
        """
        description = self.analyze_crop(crop, task_prompt="<MORE_DETAILED_CAPTION>")
        return description
