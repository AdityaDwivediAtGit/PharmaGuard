import time
from vision_detector import VisionDetector
from multimodal_reasoner import MultimodalReasoner
from llm_agent import LLMAgent
from speech_handler import SpeechHandler
from utils import time_it, get_logger

logger = get_logger(__name__)

class Pipeline:
    def __init__(self):
        logger.info("Initializing PGMI Pipeline...")
        start_time = time.time()
        self.vision = VisionDetector()
        self.vlm = MultimodalReasoner()
        self.llm = LLMAgent()
        self.speech = SpeechHandler()
        logger.info(f"Pipeline initialized in {time.time() - start_time:.2f} seconds.")
        
    @time_it
    def process(self, frame, enable_vlm=True, enable_llm=True, enable_speech=True):
        """
        Process a single frame through the entire pipeline.
        Returns the annotated frame, logs, metrics, and any audio alert.
        Allows disabling components for ablation studies.
        """
        start_time = time.time()
        logs = []
        metrics = {}
        audio_path = None
        
        # 1. Vision Stage
        v_start = time.time()
        annotated_frame, crops, metadata = self.vision.process_frame(frame)
        metrics['vision_time'] = time.time() - v_start
        
        if not crops:
            logs.append("No blister packs or tablets detected.")
            metrics['total_time'] = time.time() - start_time
            return annotated_frame, logs, metrics, audio_path

        # 2. Multimodal Reasoner Stage 
        # For real-time demo, we process the largest crop (most likely the full blister pack or a close-up)
        # Alternatively, could process all crops in batch if hardware supports it.
        largest_crop_idx = max(range(len(crops)), key=lambda i: crops[i].shape[0] * crops[i].shape[1])
        crop = crops[largest_crop_idx]
        
        vlm_desc = "VLM disabled."
        if enable_vlm:
            m_start = time.time()
            vlm_desc = self.vlm.detect_defects(crop)
            metrics['vlm_time'] = time.time() - m_start
            logs.append(f"VLM Description: {vlm_desc}")

        # 3. LLM Agent Stage
        is_defect = False
        severity = "none"
        action = "pass"
        reason = "LLM disabled."
        
        if enable_llm:
            l_start = time.time()
            assessment = self.llm.evaluate_defect(vlm_desc)
            metrics['llm_time'] = time.time() - l_start
            
            is_defect = assessment.get("defect_detected", False)
            severity = assessment.get("severity", "none")
            action = assessment.get("action", "pass")
            reason = assessment.get("reason", "No reason provided.")
            
            metrics['defect_detected'] = is_defect
            logs.append(f"Agent Action: {action.upper()} | Severity: {severity.upper()} | Reason: {reason}")

        # 4. Speech Alert Stage
        if enable_speech and is_defect and severity in ["medium", "high"]:
            s_start = time.time()
            alert_text = f"{severity} severity defect detected. Recommended action: {action}. {reason}"
            audio_path = self.speech.generate_alert(alert_text)
            metrics['speech_time'] = time.time() - s_start
            logs.append(f"Audio Alert: {alert_text}")

        metrics['total_time'] = time.time() - start_time
        
        return annotated_frame, logs, metrics, audio_path
