
"""
Utility functions for AI Text Detection Benchmark
"""

import os
import json
import yaml
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging(level: int = logging.INFO) -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def load_dataset(dataset_path: str, max_samples: Optional[int] = None) -> Tuple[List[str], List[int]]:
    """
    Load dataset from various formats
    
    Args:
        dataset_path: Path to dataset file
        max_samples: Maximum number of samples to load
        
    Returns:
        Tuple of (texts, labels)
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    if dataset_path.suffix == '.csv':
        df = pd.read_csv(dataset_path)
    elif dataset_path.suffix == '.json':
        df = pd.read_json(dataset_path)
    elif dataset_path.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(dataset_path)
    else:
        raise ValueError(f"Unsupported file format: {dataset_path.suffix}")
    
    # Assume columns are 'text' and 'label'
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns")
    
    texts = df['text'].astype(str).tolist()
    labels = df['label'].astype(int).tolist()
    
    # Limit samples if specified
    if max_samples and len(texts) > max_samples:
        texts = texts[:max_samples]
        labels = labels[:max_samples]
    
    return texts, labels

def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save configuration to YAML file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_results_directory(base_path: str) -> Path:
    """Create timestamped results directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(base_path) / f"results_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir

def plot_benchmark_comparison(results_df: pd.DataFrame, 
                            output_path: Optional[str] = None) -> None:
    """
    Create comprehensive comparison plots
    
    Args:
        results_df: DataFrame containing benchmark results
        output_path: Path to save plots
    """
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Model Benchmark Comparison', fontsize=16, fontweight='bold')
    
    # 1. Accuracy Comparison
    axes[0, 0].bar(results_df['model_name'], results_df['accuracy'])
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].set_ylim(0, 1)
    
    # 2. Inference Time Comparison
    axes[0, 1].bar(results_df['model_name'], results_df['inference_time_ms'])
    axes[0, 1].set_title('Inference Time')
    axes[0, 1].set_ylabel('Time (ms)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Model Size Comparison
    axes[0, 2].bar(results_df['model_name'], results_df['model_size_mb'])
    axes[0, 2].set_title('Model Size')
    axes[0, 2].set_ylabel('Size (MB)')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. F1 Score Comparison
    axes[1, 0].bar(results_df['model_name'], results_df['f1_score'])
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].set_ylim(0, 1)
    
    # 5. Throughput Comparison
    axes[1, 1].bar(results_df['model_name'], results_df['throughput_samples_per_sec'])
    axes[1, 1].set_title('Throughput')
    axes[1, 1].set_ylabel('Samples/Second')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 6. Memory Usage Comparison
    axes[1, 2].bar(results_df['model_name'], results_df['peak_memory_mb'])
    axes[1, 2].set_title('Peak Memory Usage')
    axes[1, 2].set_ylabel('Memory (MB)')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    plt.show()

def plot_performance_efficiency(results_df: pd.DataFrame, 
                              output_path: Optional[str] = None) -> None:
    """
    Create performance vs efficiency scatter plot
    
    Args:
        results_df: DataFrame containing benchmark results
        output_path: Path to save plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create scatter plot: Accuracy vs Inference Time
    scatter = ax.scatter(
        results_df['inference_time_ms'], 
        results_df['accuracy'], 
        s=results_df['model_size_mb'] * 3,  # Size represents model size
        alpha=0.7,
        c=results_df['f1_score'],  # Color represents F1 score
        cmap='viridis'
    )
    
    # Add labels for each point
    for i, model_name in enumerate(results_df['model_name']):
        ax.annotate(
            model_name, 
            (results_df['inference_time_ms'].iloc[i], results_df['accuracy'].iloc[i]),
            xytext=(5, 5), 
            textcoords='offset points',
            fontsize=10
        )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('F1 Score')
    
    ax.set_xlabel('Inference Time (ms)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Performance vs Efficiency\n(Bubble size = Model size, Color = F1 Score)')
    ax.grid(True, alpha=0.3)
    
    # Add optimal region
    ax.axhline(y=results_df['accuracy'].mean(), color='red', linestyle='--', alpha=0.5, label='Avg Accuracy')
    ax.axvline(x=results_df['inference_time_ms'].mean(), color='red', linestyle='--', alpha=0.5, label='Avg Inference Time')
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def calculate_efficiency_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate various efficiency metrics"""
    metrics_df = results_df.copy()
    
    # Efficiency scores
    metrics_df['accuracy_per_ms'] = metrics_df['accuracy'] / metrics_df['inference_time_ms']
    metrics_df['accuracy_per_mb'] = metrics_df['accuracy'] / metrics_df['model_size_mb']
    metrics_df['throughput_per_mb'] = metrics_df['throughput_samples_per_sec'] / metrics_df['model_size_mb']
    
    # Normalized scores (0-1 scale)
    metrics_df['accuracy_norm'] = (metrics_df['accuracy'] - metrics_df['accuracy'].min()) / (metrics_df['accuracy'].max() - metrics_df['accuracy'].min())
    metrics_df['speed_norm'] = (metrics_df['inference_time_ms'].max() - metrics_df['inference_time_ms']) / (metrics_df['inference_time_ms'].max() - metrics_df['inference_time_ms'].min())
    metrics_df['size_norm'] = (metrics_df['model_size_mb'].max() - metrics_df['model_size_mb']) / (metrics_df['model_size_mb'].max() - metrics_df['model_size_mb'].min())
    
    # Combined efficiency score
    metrics_df['efficiency_score'] = (metrics_df['accuracy_norm'] + metrics_df['speed_norm'] + metrics_df['size_norm']) / 3
    
    return metrics_df

def export_latex_table(results_df: pd.DataFrame, output_path: str) -> None:
    """Export results as LaTeX table"""
    # Select key columns
    latex_df = results_df[['model_name', 'accuracy', 'f1_score', 'inference_time_ms', 'model_size_mb', 'throughput_samples_per_sec']].copy()
    
    # Round numerical values
    latex_df['accuracy'] = latex_df['accuracy'].round(4)
    latex_df['f1_score'] = latex_df['f1_score'].round(4)
    latex_df['inference_time_ms'] = latex_df['inference_time_ms'].round(2)
    latex_df['model_size_mb'] = latex_df['model_size_mb'].round(2)
    latex_df['throughput_samples_per_sec'] = latex_df['throughput_samples_per_sec'].round(2)
    
    # Rename columns for LaTeX
    latex_df.columns = ['Model', 'Accuracy', 'F1 Score', 'Inference Time (ms)', 'Model Size (MB)', 'Throughput (samples/s)']
    
    # Export to LaTeX
    latex_str = latex_df.to_latex(index=False, float_format='%.4f')
    
    with open(output_path, 'w') as f:
        f.write(latex_str)

def generate_markdown_report(results_df: pd.DataFrame, 
                           config: Dict[str, Any],
                           output_path: str) -> None:
    """Generate markdown report"""
    
    # Calculate summary statistics
    summary_stats = {
        'total_models': len(results_df),
        'avg_accuracy': results_df['accuracy'].mean(),
        'std_accuracy': results_df['accuracy'].std(),
        'avg_inference_time': results_df['inference_time_ms'].mean(),
        'std_inference_time': results_df['inference_time_ms'].std(),
        'avg_model_size': results_df['model_size_mb'].mean(),
        'std_model_size': results_df['model_size_mb'].std(),
    }
    
    # Find best performers
    best_accuracy
