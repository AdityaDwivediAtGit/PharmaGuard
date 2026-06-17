import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import numpy as np
import cv2
from utils import time_it, get_logger, Config

logger = get_logger(__name__)

class MultimodalReasoner:
    def __init__(self, model_id="vikhyatk/moondream2"):
        # Switched to Moondream2 (1.8B). It provides highly detailed VQA without 
        # the Florence-2 transformers compatibility bugs, and is small enough for CPU.
        logger.info(f"Loading VLM model from {model_id}...")
        try:
            self.device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
            self.revision = "2024-08-26"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=self.revision)
            config = AutoConfig.from_pretrained(model_id, revision=self.revision, trust_remote_code=True)
            if not hasattr(config, 'pad_token_id') or config.pad_token_id is None:
                config.pad_token_id = config.bos_token_id if hasattr(config, 'bos_token_id') else 0
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                config=config,
                trust_remote_code=True,
                revision=self.revision,
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
    def analyze_crop(self, crop, task_prompt="Describe the condition of this pharmaceutical blister pack in detail. Are any pills missing, broken, or empty?"):
        """
        Analyze a cropped image using Moondream2 VQA.
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

        # Moondream2 specific inference
        enc_image = self.model.encode_image(image)
        answer = self.model.answer_question(enc_image, task_prompt, self.tokenizer)
        
        return answer

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
