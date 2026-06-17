import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import Config, get_logger

logger = get_logger(__name__)

def generate_charts():
    csv_file = os.path.join(Config.BASE_DIR, "advanced_ablations.csv")
    if not os.path.exists(csv_file):
        logger.error("advanced_ablations.csv not found. Run advanced_ablations.py first.")
        return

    df = pd.read_csv(csv_file)
    os.makedirs(os.path.join(Config.BASE_DIR, "charts"), exist_ok=True)
    
    # Set seaborn style for premium look
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 6), 'figure.dpi': 300})

    # 1. Pipeline Depth - Latency Breakdown (Stacked Bar)
    depth_df = df[df['Study'] == 'Pipeline Depth'].copy()
    depth_df.set_index('Configuration', inplace=True)
    
    ax = depth_df[['Vision_ms', 'VLM_ms', 'LLM_ms']].plot(kind='bar', stacked=True, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title('Latency Breakdown by Pipeline Depth', fontweight='bold')
    plt.ylabel('Latency (ms)')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.BASE_DIR, "charts", "pipeline_depth_latency.png"))
    plt.close()

    # 2. Resolution Scaling - FPS vs F1 Score (Line + Scatter)
    res_df = df[df['Study'] == 'Resolution Scaling'].copy()
    fig, ax1 = plt.subplots()
    
    color = 'tab:red'
    ax1.set_xlabel('Resolution')
    ax1.set_ylabel('FPS', color=color)
    ax1.plot(res_df['Resolution'], res_df['FPS'], marker='o', color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Est. F1 Score', color=color)  
    ax2.plot(res_df['Resolution'], res_df['Est_F1_Score'], marker='s', color=color, linestyle='--', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('FPS & Accuracy Trade-off across Resolutions', fontweight='bold')
    fig.tight_layout()  
    plt.savefig(os.path.join(Config.BASE_DIR, "charts", "resolution_tradeoff.png"))
    plt.close()

    # 3. Quantization - Memory vs Latency (Bubble Chart)
    quant_df = df[df['Study'] == 'Quantization'].copy()
    plt.figure(figsize=(10, 6))
    
    scatter = sns.scatterplot(
        data=quant_df, 
        x='Total_Latency_ms', 
        y='GPU_Mem_MB', 
        hue='Configuration',
        size='Est_F1_Score', 
        sizes=(100, 500), 
        alpha=0.7, 
        palette="viridis"
    )
    
    plt.title('Hardware Efficiency: ROCm FP8 & Quantization Impact on MI300X', fontweight='bold')
    plt.xlabel('Total Latency (ms)')
    plt.ylabel('GPU Memory Usage (MB)')
    
    # Move legend outside
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.BASE_DIR, "charts", "quantization_efficiency.png"))
    plt.close()

    logger.info("Successfully generated publication-ready charts in the 'charts/' directory.")

if __name__ == "__main__":
    generate_charts()
