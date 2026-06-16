# PharmaGuard Multimodal Inspector (PGMI)

This document outlines the implementation plan for the **PharmaGuard Multimodal Inspector (PGMI)**, a production-grade prototype application for real-time quality control of pharmaceutical blister packaging lines, optimized for AMD MI300X with ROCm 7.2.

## Goal Description

Build an end-to-end multimodal pipeline that leverages Vision (YOLO/Florence), Multimodal Reasoning (VLM like Qwen2.5-VL), and LLM-based Agentic decision-making (Llama-3.1/Qwen2.5) to detect defects in blister packs (broken tablets, empty pockets, foreign particles). The pipeline will provide real-time visual overlays, text explanations, and speech alerts, presented through a Gradio UI. The focus is on low latency (<2s per frame), low memory footprint (via 4-bit quantization), and AMD ROCm optimizations.

## User Review Required

> [!IMPORTANT]
> - **Hardware Constraints**: The application is heavily optimized for AMD ROCm 7.2. During development on this environment, fallbacks to CPU or standard CUDA might be necessary if we cannot run ROCm natively. I will implement CPU fallbacks for demo purposes if ROCm is not detected.
> - **Model Selection**: Due to hardware limitations and download times, downloading full 7B models (like Qwen2.5-VL-7B and Llama-3.1-8B) might take time or fail if the environment doesn't have sufficient resources. I will provide scripts to download them, but for immediate testing, we might need to rely on smaller models or mock implementations if full models cannot run locally.

## Open Questions

> [!WARNING]
> 1. Do you have a Hugging Face token ready with access to gated models (like Llama-3.1-8B-Instruct)? If so, you will need to log in via `huggingface-cli login` before running the pipeline.
> 2. For the demo data, should I write a script to automatically download some sample blister pack images from a public dataset (like Roboflow), or will you provide specific images in the `data/` folder? I will include placeholder code to generate synthetic images if none are provided.

## Proposed Architecture & Changes

We will create the `pharmaguard/` directory structure with the following components:

### Foundation and Utilities
- `requirements.txt`: Dependencies including PyTorch for ROCm, Transformers, Optimum-AMD, Gradio, etc.
- `README.md`: Setup instructions, ROCm notes, and hackathon details.
- `utils.py`: Logging, timing decorators for benchmarking, and configuration constants.

### Vision Stage
- `vision_detector.py`: Implements a fast object detection pipeline using YOLOv11 (Ultralytics) to detect blister packs and tablets. Includes basic tracking to maintain identity across simulated frames.

### Multimodal Stage
- `multimodal_reasoner.py`: Integrates a Vision-Language Model (VLM, targeting Qwen2.5-VL-7B-Instruct or Florence-2 as fallback) to analyze cropped images of detected packs and describe specific defects.

### LLM Agent Stage
- `llm_agent.py`: Uses a fast LLM (Llama-3.1-8B or Qwen2.5-7B) to process the VLM's description, assess severity, check compliance, and recommend actions (e.g., divert, stop line) outputting structured JSON.

### Speech and Alerts
- `speech_handler.py`: Implements Text-to-Speech (TTS) using `gTTS` or `piper-tts` to generate spoken alerts for high-severity defects.

### Orchestration and UI
- `pipeline.py`: The core orchestrator that ties Vision -> VLM -> LLM -> Speech together, ensuring data flows correctly and timing is recorded.
- `app.py`: The Gradio interface featuring tabs for Live Demo, Metrics, Logs, and About.

### Data and Assets
- `data/` & `demo_video_assets/`: Directories for sample inputs.

## Verification Plan

### Automated Tests
- Running `python pipeline.py` to test the end-to-end flow on a single mock image.
- Using the "Benchmark" feature in the Gradio app to simulate 50 frames and ensure latency is < 2s on average (hardware dependent).

### Manual Verification
- Launch the Gradio app (`python app.py`) and verify the UI components (overlays, defect list, speech playback, and metrics).
- Test the system with an image containing an obvious defect (e.g., missing pill) to verify that the LLM Agent correctly flags it as high severity and recommends diversion.
