import gradio as gr
import cv2
import numpy as np
import os
import time
from pipeline import Pipeline
from utils import get_logger

logger = get_logger(__name__)

# Initialize pipeline globally to avoid reloading models on every request
try:
    pgmi_pipeline = Pipeline()
except Exception as e:
    logger.error(f"Failed to initialize pipeline: {e}")
    pgmi_pipeline = None

def process_image(image):
    if pgmi_pipeline is None:
        return image, "Pipeline not initialized.", {"error": "Initialization failed"}, None
        
    if image is None:
        return None, "No image provided.", {}, None
        
    # image is a numpy array (RGB from Gradio)
    # Convert to BGR for OpenCV processing inside pipeline
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    annotated_bgr, logs, metrics, audio_path = pgmi_pipeline.process(bgr_image)
    
    # Convert back to RGB for Gradio display
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    
    log_text = "\n".join(logs)
    return annotated_rgb, log_text, metrics, audio_path

def benchmark():
    if pgmi_pipeline is None:
        return "Pipeline not initialized.", {}
        
    # Create a dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    logs_agg = []
    latencies = []
    
    # Run 5 frames for benchmark
    for i in range(5): 
        start = time.time()
        _, logs, _, _ = pgmi_pipeline.process(dummy_image)
        latencies.append(time.time() - start)
        logs_agg.extend(logs)
        
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    
    result = f"Ran benchmark over {len(latencies)} frames.\nAverage Latency: {avg_latency:.2f}s per frame.\nEstimated FPS: {1/avg_latency if avg_latency > 0 else 0:.2f}"
    
    # Exact memory usage from torch
    import torch
    mem_used = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    
    metrics_data = {
        "Avg Latency (s)": round(avg_latency, 2),
        "Peak GPU Mem (MB)": round(mem_used, 2)
    }
    
    return result, metrics_data

with gr.Blocks(title="PharmaGuard Multimodal Inspector", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# PharmaGuard Multimodal Inspector (PGMI)")
    gr.Markdown("Real-time quality control for pharmaceutical blister packaging using AMD MI300X ROCm.")
    
    with gr.Tabs():
        with gr.Tab("Live Demo"):
            with gr.Row():
                with gr.Column(scale=2):
                    input_image = gr.Image(label="Input Frame", sources=["upload", "webcam"])
                    process_btn = gr.Button("Inspect Frame", variant="primary")
                with gr.Column(scale=3):
                    output_image = gr.Image(label="Annotated Frame")
                    audio_output = gr.Audio(label="Speech Alert", autoplay=True)
            
            with gr.Row():
                log_output = gr.Textbox(label="Agent Logs & Reasoning", lines=5)
                metrics_output = gr.JSON(label="Processing Metrics")
                
            process_btn.click(
                fn=process_image,
                inputs=input_image,
                outputs=[output_image, log_output, metrics_output, audio_output]
            )
            
        with gr.Tab("Metrics & Benchmark"):
            gr.Markdown("### Run performance benchmark on simulated data")
            bench_btn = gr.Button("Run Benchmark (5 Frames)")
            bench_text = gr.Textbox(label="Benchmark Results")
            bench_json = gr.JSON(label="Hardware Metrics")
            
            bench_btn.click(
                fn=benchmark,
                inputs=None,
                outputs=[bench_text, bench_json]
            )
            
            gr.Markdown("### Model Insights")
            gr.Markdown("""
            | Component | Model | Parameters | Quantization | Framework |
            |-----------|-------|------------|--------------|-----------|
            | Vision | YOLOv11n | 2.6M | None | Ultralytics/PyTorch |
            | VLM | Florence-2-large | 0.77B | None | Transformers |
            | LLM Agent | Qwen2.5-1.5B-Instruct | 1.5B | 4-bit (bitsandbytes) | Transformers |
            | Speech | gTTS | N/A | None | Google Translate API |
            """)
            
        with gr.Tab("About"):
            gr.Markdown("""
            ### TCS & AMD AI Hackathon 2026 - Track 2 Multimodal
            
            **PharmaGuard Multimodal Inspector (PGMI)** is designed to perform zero-shot and few-shot defect detection on high-speed blister packaging lines.
            
            By fusing traditional fast object detection (YOLO) with deep visual reasoning (VLM) and agentic decision-making (LLM), PGMI can intelligently divert defective packs while explaining *why* it made that decision, directly to human operators via text and speech.
            
            Optimized specifically for **AMD MI300X** utilizing **ROCm 7.2**.
            """)

if __name__ == "__main__":
    start_port = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
    share_mode = os.environ.get("GRADIO_SHARE", "false").lower() in ("1", "true", "yes")
    if share_mode:
        logger.info("GRADIO_SHARE enabled: Gradio will create a public share link.")

    for port in range(start_port, start_port + 10):
        try:
            local_url = f"http://127.0.0.1:{port}"
            logger.info(f"Starting Gradio on port {port}...")
            print(f"Gradio app available at: {local_url}")
            if share_mode:
                print("Public Gradio share enabled. Waiting for public link...")

            launch_result = demo.launch(
                server_name="0.0.0.0",
                server_port=port,
                share=share_mode,
                prevent_thread_lock=True,
            )

            if share_mode:
                try:
                    if isinstance(launch_result, tuple) and len(launch_result) >= 3:
                        public_url = launch_result[2]
                    else:
                        public_url = getattr(launch_result, 'share_url', None)
                    if public_url:
                        print(f"Public Gradio link: {public_url}")
                except Exception:
                    logger.warning("Unable to print Gradio public share URL from launch result.")

            break
        except OSError as e:
            logger.warning(f"Port {port} unavailable: {e}")
            if port == start_port + 9:
                raise
