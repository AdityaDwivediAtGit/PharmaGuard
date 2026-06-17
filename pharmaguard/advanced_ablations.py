import os
import cv2
import csv
import json
import torch
import subprocess
from sklearn.metrics import f1_score
from pipeline import Pipeline
from download_data import create_eval_dataset
from utils import Config, get_logger

logger = get_logger(__name__)

def get_gpu_memory():
    """Returns GPU max memory allocated in MB if CUDA is available."""
    if torch.cuda.is_available():
        # Reset peak stats before measurement to get delta
        torch.cuda.reset_peak_memory_stats()
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0

def get_rocm_info():
    """Try to get ROCm SMI info."""
    try:
        result = subprocess.run(['rocm-smi', '--showproductname'], capture_output=True, text=True)
        return result.stdout.strip().replace('\n', ' | ')
    except Exception:
        return "ROCm SMI not available or not in PATH."

def load_dataset():
    eval_dir = os.path.join(Config.DATA_DIR, "eval_set")
    labels_path = os.path.join(eval_dir, "labels.json")
    if not os.path.exists(labels_path):
        create_eval_dataset()
    
    with open(labels_path, "r") as f:
        labels = json.load(f)
        
    dataset = []
    for filename, data in labels.items():
        img_path = os.path.join(eval_dir, filename)
        img = cv2.imread(img_path)
        if img is not None:
            dataset.append({"img": img, "label": data["defect_detected"], "filename": filename})
            
    return dataset

def evaluate_pipeline(pgmi, dataset, config_name, enable_vlm=True, enable_llm=True, enable_speech=False, resolution_name="480p"):
    y_true = []
    y_pred = []
    
    total_latencies = []
    v_latencies = []
    m_latencies = []
    l_latencies = []
    s_latencies = []
    
    # Warmup
    if dataset:
        pgmi.process(dataset[0]["img"].copy(), enable_vlm, enable_llm, enable_speech)
        
    start_mem = get_gpu_memory()
    
    for item in dataset:
        img = item["img"].copy()
        
        # Apply resolution scaling if needed
        if resolution_name == "720p":
            img = cv2.resize(img, (1280, 720))
        elif resolution_name == "1080p":
            img = cv2.resize(img, (1920, 1080))
        elif resolution_name == "4K":
            img = cv2.resize(img, (3840, 2160))
            
        y_true.append(item["label"])
        
        _, logs, metrics, _ = pgmi.process(img, enable_vlm, enable_llm, enable_speech)
        
        total_latencies.append(metrics.get("total_time", 0))
        v_latencies.append(metrics.get("vision_time", 0))
        if enable_vlm: m_latencies.append(metrics.get("vlm_time", 0))
        if enable_llm: l_latencies.append(metrics.get("llm_time", 0))
        if enable_speech: s_latencies.append(metrics.get("speech_time", 0))
        
        # Exact strict evaluation
        if enable_llm:
            pred = metrics.get("defect_detected", False)
        elif enable_vlm:
            # Simple heuristic if VLM is on but LLM is off
            desc = logs[-1].lower() if logs else ""
            pred = "empty" in desc or "broken" in desc or "missing" in desc
        else:
            # Vision only heuristic (in our prototype YOLO just tracks boxes, doesn't classify missing well yet)
            # If a bounding box is found, it's NOT defective (simplified for strict eval without mock). 
            # If no boxes, maybe defective.
            pred = len(logs) > 0 and "No blister packs" in logs[0]
            
        y_pred.append(pred)

    end_mem = get_gpu_memory()
    mem_used = max(0, end_mem - start_mem)
    
    # Calculate exact F1 Score
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    avg_total = sum(total_latencies) / len(dataset) if dataset else 0
    avg_v = sum(v_latencies) / len(dataset) if dataset else 0
    avg_m = sum(m_latencies) / len(dataset) if dataset else 0
    avg_l = sum(l_latencies) / len(dataset) if dataset else 0
    avg_s = sum(s_latencies) / len(dataset) if dataset else 0
    
    return {
        "Configuration": config_name,
        "Resolution": resolution_name,
        "Total_Latency_ms": round(avg_total * 1000, 2),
        "FPS": round(1.0 / avg_total if avg_total > 0 else 0, 1),
        "GPU_Mem_MB": round(mem_used, 2),
        "Vision_ms": round(avg_v * 1000, 2),
        "VLM_ms": round(avg_m * 1000, 2),
        "LLM_ms": round(avg_l * 1000, 2),
        "Speech_ms": round(avg_s * 1000, 2),
        "Est_F1_Score": round(f1, 2)
    }

def run_advanced_experiments():
    logger.info("Initializing Strict Measurement Pipeline...")
    
    dataset = load_dataset()
    if not dataset:
        logger.error("Dataset could not be loaded or generated.")
        return
        
    pgmi = Pipeline()
    gpu_info = get_rocm_info()
    logger.info(f"Hardware Detected: {gpu_info}")

    results = []
    
    # --- Study 1: Pipeline Depth Impact ---
    logger.info("--- Running Study 1: Pipeline Depth (Exact Measurements) ---")
    
    res1 = evaluate_pipeline(pgmi, dataset, "Vision Only", enable_vlm=False, enable_llm=False, enable_speech=False)
    res1["Study"] = "Pipeline Depth"
    results.append(res1)
    
    res2 = evaluate_pipeline(pgmi, dataset, "Vision + VLM", enable_vlm=True, enable_llm=False, enable_speech=False)
    res2["Study"] = "Pipeline Depth"
    results.append(res2)
    
    res3 = evaluate_pipeline(pgmi, dataset, "Vision + VLM + LLM Reasoner", enable_vlm=True, enable_llm=True, enable_speech=False)
    res3["Study"] = "Pipeline Depth"
    results.append(res3)
    
    res4 = evaluate_pipeline(pgmi, dataset, "Full Pipeline (+Speech)", enable_vlm=True, enable_llm=True, enable_speech=True)
    res4["Study"] = "Pipeline Depth"
    results.append(res4)

    # --- Study 2: Resolution Scaling ---
    logger.info("--- Running Study 2: Resolution Scaling Impact (Exact Measurements) ---")
    
    for res in ["480p", "720p", "1080p", "4K"]:
        r = evaluate_pipeline(pgmi, dataset, "Full Pipeline", enable_vlm=True, enable_llm=True, enable_speech=False, resolution_name=res)
        r["Study"] = "Resolution Scaling"
        results.append(r)

    # --- Study 3: Quantization (Strict Evaluation) ---
    logger.info("--- Running Study 3: Quantization (Exact Measurements) ---")
    # For a completely strict measurement, we cannot simulate quantization.
    # We log the current state (which loads in 4-bit by default in llm_agent.py if bitsandbytes is present).
    # Since dynamic unloading/reloading of PyTorch models safely within a single script
    # can cause OOM on limited hardware, we measure the current loaded state.
    
    q_res = evaluate_pipeline(pgmi, dataset, "Current Loaded Config (4-bit if supported)", enable_vlm=True, enable_llm=True, enable_speech=False)
    q_res["Study"] = "Quantization"
    results.append(q_res)

    # Save CSV
    csv_file = os.path.join(Config.BASE_DIR, "advanced_ablations.csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Strict experiments completed. Exact results saved to {csv_file}")

if __name__ == "__main__":
    run_advanced_experiments()
