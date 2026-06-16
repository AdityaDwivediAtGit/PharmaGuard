import json
import torch
from transformers import pipeline
from utils import time_it, get_logger, Config

logger = get_logger(__name__)

class LLMAgent:
    def __init__(self, model_id=Config.LLM_MODEL_ID):
        logger.info(f"Loading LLM Agent from {model_id}...")
        try:
            self.device = 0 if torch.cuda.is_available() else -1
            
            # Use 4-bit quantization if bitsandbytes is available and running on CUDA
            model_kwargs = {}
            if self.device == 0:
                model_kwargs["load_in_4bit"] = True
                
            self.generator = pipeline(
                "text-generation", 
                model=model_id, 
                device_map="auto", # auto device mapping handles quantization better
                torch_dtype=torch.float16 if self.device == 0 else torch.float32,
                model_kwargs=model_kwargs
            )
            logger.info("LLM Agent loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load LLM Agent: {e}")
            self.generator = None

    @time_it
    def evaluate_defect(self, vlm_description):
        """
        Takes the description from the VLM and evaluates severity and action.
        Returns a JSON object.
        """
        if self.generator is None:
            return {"defect_detected": False, "severity": "none", "action": "pass", "reason": "LLM not loaded"}

        prompt = f"""<|im_start|>system
You are a Quality Control AI for a pharmaceutical blister packaging line. 
Based on the following visual description of a blister pack crop, determine if there is a defect.
A defect includes: broken tablets, cracked tablets, empty pockets, foreign particles, or color mismatches.

Respond ONLY with strict JSON in the following format:
{{
    "defect_detected": true/false,
    "severity": "low/medium/high/none",
    "action": "divert/stop_line/pass",
    "reason": "short explanation"
}}
<|im_end|>
<|im_start|>user
Visual Description: {vlm_description}
<|im_end|>
<|im_start|>assistant
"""
        try:
            response = self.generator(prompt, max_new_tokens=150, return_full_text=False)[0]['generated_text']
            # Clean up response to find JSON
            json_str = response[response.find('{'):response.rfind('}')+1]
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"LLM Agent failed to parse output: {e}\nRaw output: {response if 'response' in locals() else 'N/A'}")
            # Fallback heuristic if LLM fails JSON
            if "empty" in vlm_description.lower() or "broken" in vlm_description.lower() or "missing" in vlm_description.lower():
                return {"defect_detected": True, "severity": "high", "action": "divert", "reason": "Heuristic fallback: potential defect"}
            return {"defect_detected": False, "severity": "unknown", "action": "manual_review", "reason": "Parsing failed"}
