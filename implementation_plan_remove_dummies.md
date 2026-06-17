# Refactoring Plan: Removing Mock Data & Enforcing Strict Measurement Standards

Based on your request, I will refactor the codebase to strictly eliminate any hardcoded "dummy" or "simulated" performance numbers. Everything will be measured in real-time using PyTorch and standard evaluation metrics (like `sklearn.metrics` for F1). 

> [!WARNING]
> Because we are removing the safety fallback simulations, **if your AMD environment runs out of GPU memory (OOM) or a model fails to download, the scripts will now accurately report 0.0 FPS or throw an error.** You will need to ensure your `notebooks.amd.com` container has sufficient VRAM to load Florence-2 and Llama/Qwen simultaneously.

## Proposed Changes

### 1. `app.py` (Gradio UI)
- **[MODIFY]**: Remove the hardcoded `Peak GPU Mem (GB): ~12.4` and `Est. Accuracy F1: 95.2%` in the `benchmark()` function.
- **[MODIFY]**: Update `benchmark()` to query actual GPU memory usage using `torch.cuda.max_memory_allocated()`. Accuracy will be omitted from the live dashboard since F1 cannot be calculated on a single unlabelled frame; it will be restricted to the dedicated experiment scripts.

### 2. `run_experiments.py` & `advanced_ablations.py`
- **[MODIFY]**: Delete the `MI300X_BASELINES` simulation dictionary.
- **[MODIFY]**: Remove all `if not models_loaded:` fallback logic that injected realistic MI300X numbers.
- **[MODIFY]**: Implement a true ground-truth evaluation function. I will update `download_data.py` to generate a mini dataset of 10 images (5 normal, 5 defective) with an accompanying `labels.json`.
- **[MODIFY]**: The scripts will run inference on this dataset and calculate the *exact* precision, recall, and F1 score based purely on the LLM's actual parsed outputs.
- **[MODIFY]**: For the **Quantization** study, instead of applying a mathematical multiplier (e.g., 0.45x for INT4), the script will explicitly unload the model, reload it with `load_in_4bit=True`, run inference, and measure the real physical latency and memory footprint. *(Note: We will only test FP16 and INT4, as FP8 requires specific hardware/library flags that might crash standard transformers without explicit optimum-amd compilation).*

### 3. `business_impact.py`
- **[MODIFY]**: The calculator will ingest the *real* F1 score and *real* FPS from your actual hardware run. 
- **[MODIFY]**: While production volumes (e.g., 300 packs/min) must inherently be assumed constants for a business equation, they will be moved to the top of the script as explicitly defined "Business Assumptions" rather than being hidden in the code.

## Verification Plan
1. Run `python download_data.py` to generate the labeled mini-dataset.
2. Run `python advanced_ablations.py` and verify it uses `torch.cuda` memory profiling and outputs exact latencies and a mathematically sound F1 score.
3. Review the generated `.csv` to confirm no simulated numbers exist.

> [!IMPORTANT]
> Do you approve removing all simulation fallbacks? Please confirm, and I will proceed with rewriting the scripts to be 100% strictly measured.
