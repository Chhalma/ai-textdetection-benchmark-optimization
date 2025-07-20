
"""
Core Evaluator Module for AI Text Detection Benchmark
Provides comprehensive model evaluation and benchmarking capabilities.
"""

import os
import time
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    pipeline, PretrainedConfig
)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
import psutil
import threading
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkMetrics:
    """Data class for storing benchmark results"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    inference_time_ms: float
    memory_usage_mb: float
    model_size_mb: float
    throughput_samples_per_sec: float
    peak_memory_mb: float
    gpu_memory_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class MemoryMonitor:
    """Monitor memory usage during evaluation"""
    
    def __init__(self):
        self.peak_memory = 0
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start memory monitoring in a separate thread"""
        self.monitoring = True
        self.peak_memory = psutil.virtual_memory().used / (1024**2)
        self.monitor_thread = threading.Thread(target=self._monitor_memory)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self) -> float:
        """Stop monitoring and return peak memory usage"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        return self.peak_memory
        
    def _monitor_memory(self):
        """Internal memory monitoring loop"""
        while self.monitoring:
            current_memory = psutil.virtual_memory().used / (1024**2)
            self.peak_memory = max(self.peak_memory, current_memory)
            time.sleep(0.1)

class BenchmarkRunner:
    """Main class for running comprehensive model benchmarks"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the benchmark runner
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.device = self._setup_device()
        self.results_dir = Path(self.config.get('output', {}).get('results_dir', 'results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage for results
        self.all_results: List[BenchmarkMetrics] = []
        self.detailed_results: Dict[str, Any] = {}
        
        logger.info(f"BenchmarkRunner initialized with device: {self.device}")
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'models': {
                    'bert_base': {
                        'model_name': 'bert-base-uncased',
                        'max_length': 512,
                        'batch_size': 16
                    }
                },
                'evaluation': {
                    'metrics': ['accuracy', 'precision', 'recall', 'f1_score'],
                    'benchmark_samples': 1000
                },
                'output': {
                    'results_dir': 'results',
                    'save_models': True,
                    'save_plots': True
                }
            }
    
    def _setup_device(self) -> torch.device:
        """Setup computation device"""
        hardware_config = self.config.get('hardware', {})
        device_preference = hardware_config.get('device', 'auto')
        
        if device_preference == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device_preference == 'cuda':
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                logger.warning("CUDA not available, falling back to CPU")
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')
            
        return device
    
    def evaluate_model(self, 
                      model_name: str, 
                      model_config: Dict[str, Any],
                      test_texts: List[str],
                      test_labels: List[int]) -> Optional[BenchmarkMetrics]:
        """
        Evaluate a single model comprehensively
        
        Args:
            model_name: Name identifier for the model
            model_config: Configuration dictionary for the model
            test_texts: List of test text samples
            test_labels: List of corresponding labels
            
        Returns:
            BenchmarkMetrics object with evaluation results
        """
        logger.info(f"Evaluating model: {model_name}")
        
        try:
            # Load model and tokenizer
            tokenizer, model = self._load_model_and_tokenizer(model_config)
            if model is None:
                logger.error(f"Failed to load model: {model_name}")
                return None
            
            # Move model to device
            model = model.to(self.device)
            
            # Calculate model size
            model_size_mb = self._calculate_model_size(model)
            
            # Setup memory monitoring
            memory_monitor = MemoryMonitor()
            memory_monitor.start_monitoring()
            
            # Measure inference time and accuracy
            start_time = time.time()
            predictions, avg_inference_time = self._evaluate_predictions(
                model, tokenizer, test_texts, model_config
            )
            end_time = time.time()
            
            # Stop memory monitoring
            peak_memory_mb = memory_monitor.stop_monitoring()
            
            # Calculate metrics
            accuracy = accuracy_score(test_labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                test_labels, predictions, average='weighted'
            )
            
            # Calculate throughput
            total_time = end_time - start_time
            throughput = len(test_texts) / total_time if total_time > 0 else 0
            
            # Get current memory usage
            current_memory_mb = psutil.virtual_memory().used / (1024**2)
            
            # Get GPU memory if available
            gpu_memory_mb = 0.0
            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
                torch.cuda.reset_peak_memory_stats()
            
            # Create benchmark metrics
            metrics = BenchmarkMetrics(
                model_name=model_name,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                inference_time_ms=avg_inference_time,
                memory_usage_mb=current_memory_mb,
                model_size_mb=model_size_mb,
                throughput_samples_per_sec=throughput,
                peak_memory_mb=peak_memory_mb,
                gpu_memory_mb=gpu_memory_mb
            )
            
            # Store detailed results
            self.detailed_results[model_name] = {
                'predictions': predictions,
                'true_labels': test_labels,
                'classification_report': classification_report(test_labels, predictions),
                'confusion_matrix': confusion_matrix(test_labels, predictions).tolist(),
                'config': model_config,
                'evaluation_time': datetime.now().isoformat()
            }
            
            # Clean up
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"✓ {model_name} evaluation completed")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
            return None
    
    def _load_model_and_tokenizer(self, model_config: Dict[str, Any]) -> Tuple[Any, Any]:
        """Load model and tokenizer from configuration"""
        model_name = model_config['model_name']
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2,
                problem_type="single_label_classification"
            )
            return tokenizer, model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            return None, None
    
    def _calculate_model_size(self, model: torch.nn.Module) -> float:
        """Calculate model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)
    
    def _evaluate_predictions(self, 
                             model: torch.nn.Module, 
                             tokenizer: Any,
                             test_texts: List[str],
                             model_config: Dict[str, Any]) -> Tuple[List[int], float]:
        """Evaluate model predictions and measure inference time"""
        
        batch_size = model_config.get('batch_size', 16)
        max_length = model_config.get('max_length', 512)
        
        # Create classifier pipeline
        classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=0 if self.device.type == 'cuda' else -1,
            batch_size=batch_size,
            max_length=max_length,
            truncation=True,
            padding=True
        )
        
        predictions = []
        inference_times = []
        
        # Process in batches
        for i in range(0, len(test_texts), batch_size):
            batch_texts = test_texts[i:i + batch_size]
            
            # Measure inference time for this batch
            start_time = time.time()
            batch_predictions = classifier(batch_texts)
            end_time = time.time()
            
            # Record timing
            batch_time = (end_time - start_time) * 1000  # Convert to ms
            inference_times.append(batch_time / len(batch_texts))  # Per sample
            
            # Convert predictions to labels
            batch_labels = []
            for pred in batch_predictions:
                if isinstance(pred, dict):
                    # Handle HuggingFace pipeline output
                    if pred['label'] in ['LABEL_1', 'POSITIVE', '1']:
                        batch_labels.append(1)
                    else:
                        batch_labels.append(0)
                else:
                    batch_labels.append(int(pred))
            
            predictions.extend(batch_labels)
        
        avg_inference_time = np.mean(inference_times)
        return predictions, avg_inference_time
    
    def run_benchmark_suite(self, 
                           test_texts: List[str], 
                           test_labels: List[int],
                           models_to_test: Optional[List[str]] = None) -> List[BenchmarkMetrics]:
        """
        Run benchmark suite on multiple models
        
        Args:
            test_texts: List of test text samples
            test_labels: List of corresponding labels
            models_to_test: List of model names to test (None for all)
            
        Returns:
            List of BenchmarkMetrics for all evaluated models
        """
        logger.info("Starting benchmark suite")
        
        # Determine which models to test
        available_models = self.config.get('models', {})
        if models_to_test is None:
            models_to_test = list(available_models.keys())
        
        # Validate test data
        if len(test_texts) != len(test_labels):
            raise ValueError("Test texts and labels must have the same length")
        
        # Run evaluations
        results = []
        for model_name in models_to_test:
            if model_name not in available_models:
                logger.warning(f"Model {model_name} not found in configuration")
                continue
            
            model_config = available_models[model_name]
            metrics = self.evaluate_model(
                model_name, model_config, test_texts, test_labels
            )
            
            if metrics:
                results.append(metrics)
                self.all_results.append(metrics)
        
        logger.info(f"Benchmark suite completed. Evaluated {len(results)} models.")
        return results
    
    def save_results(self, results: List[BenchmarkMetrics], 
                    save_detailed: bool = True) -> str:
        """
        Save benchmark results to files
        
        Args:
            results: List of BenchmarkMetrics to save
            save_detailed: Whether to save detailed results
            
        Returns:
            Path to saved results file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory structure
        csv_dir = self.results_dir / "csv"
        json_dir = self.results_dir / "json"
        csv_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary results as CSV
        results_df = pd.DataFrame([r.to_dict() for r in results])
        csv_path = csv_dir / f"benchmark_results_{timestamp}.csv"
        results_df.to_csv(csv_path, index=False)
        
        # Save detailed results as JSON
        if save_detailed:
            detailed_path = json_dir / f"detailed_results_{timestamp}.json"
            with open(detailed_path, 'w') as f:
                json.dump(self.detailed_results, f, indent=2, default=str)
        
        # Save configuration
        config_path = json_dir / f"config_{timestamp}.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Results saved to: {csv_path}")
        return str(csv_path)
    
    def generate_comparison_report(self, results: List[BenchmarkMetrics]) -> str:
        """
        Generate a comprehensive comparison report
        
        Args:
            results: List of BenchmarkMetrics to compare
            
        Returns:
            Formatted comparison report as string
        """
        if not results:
            return "No results to compare"
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([r.to_dict() for r in results])
        
        # Find best performing models
        best_accuracy = df.loc[df['accuracy'].idxmax()]
        best_speed = df.loc[df['inference_time_ms'].idxmin()]
        best_efficiency = df.loc[(df['accuracy'] / df['inference_time_ms']).idxmax()]
        smallest_model = df.loc[df['model_size_mb'].idxmin()]
        
        # Generate report
        report = f"""
{'='*60}
BENCHMARK COMPARISON REPORT
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY STATISTICS:
{'-'*30}
Total Models Evaluated: {len(results)}
Average Accuracy: {df['accuracy'].mean():.4f} (±{df['accuracy'].std():.4f})
Average Inference Time: {df['inference_time_ms'].mean():.2f}ms (±{df['inference_time_ms'].std():.2f})
Average Model Size: {df['model_size_mb'].mean():.2f}MB (±{df['model_size_mb'].std():.2f})

BEST PERFORMERS:
{'-'*30}
🎯 Best Accuracy: {best_accuracy['model_name']} ({best_accuracy['accuracy']:.4f})
⚡ Fastest Model: {best_speed['model_name']} ({best_speed['inference_time_ms']:.2f}ms)
💾 Smallest Model: {smallest_model['model_name']} ({smallest_model['model_size_mb']:.2f}MB)
⚖️ Most Efficient: {best_efficiency['model_name']} (acc/time: {best_efficiency['accuracy']/best_efficiency['inference_time_ms']:.4f})

DETAILED RESULTS:
{'-'*30}
{df.to_string(index=False)}

PERFORMANCE RANKINGS:
{'-'*30}
By Accuracy:
{df.nlargest(len(df), 'accuracy')[['model_name', 'accuracy']].to_string(index=False)}

By Speed (Inference Time):
{df.nsmallest(len(df), 'inference_time_ms')[['model_name', 'inference_time_ms']].to_string(index=False)}

By Model Size:
{df.nsmallest(len(df), 'model_size_mb')[['model_name', 'model_size_mb']].to_string(index=False)}

RECOMMENDATIONS:
{'-'*30}
• For highest accuracy: Use {best_accuracy['model_name']}
• For fastest inference: Use {best_speed['model_name']}
• For mobile/edge deployment: Consider {smallest_model['model_name']}
• For balanced performance: Consider {best_efficiency['model_name']}

{'='*60}
        """
        
        return report.strip()
    
    def export_for_optimization(self, model_name: str, 
                               export_path: Optional[str] = None) -> str:
        """
        Export model configuration for optimization experiments
        
        Args:
            model_name: Name of the model to export
            export_path: Path to save the exported configuration
            
        Returns:
            Path to exported configuration file
        """
        if model_name not in self.config.get('models', {}):
            raise ValueError(f"Model {model_name} not found in configuration")
        
        model_config = self.config['models'][model_name]
        
        # Get results for this model if available
        model_results = None
        for result in self.all_results:
            if result.model_name == model_name:
                model_results = result.to_dict()
                break
        
        export_config = {
            'model_name': model_name,
            'model_config': model_config,
            'baseline_results': model_results,
            'export_timestamp': datetime.now().isoformat(),
            'optimization_targets': {
                'quantization': True,
                'pruning': True,
                'onnx_conversion': True
            }
        }
        
        if export_path is None:
            export_path = self.results_dir / f"{model_name}_optimization_config.json"
        
        with open(export_path, 'w') as f:
            json.dump(export_config, f, indent=2)
        
        logger.info(f"Optimization config exported to: {export_path}")
        return str(export_path)
    
    def cleanup(self):
        """Clean up resources"""
        # Clear results
        self.all_results.clear()
        self.detailed_results.clear()
        
        # Clear GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("BenchmarkRunner cleanup completed")

# Utility functions for standalone usage
def create_sample_dataset(size: int = 1000) -> Tuple[List[str], List[int]]:
    """Create a sample dataset for testing"""
    human_texts = [
        "The concept of artificial intelligence has evolved significantly over the past decade.",
        "Climate change continues to be one of the most pressing global challenges.",
        "The history of literature reveals fascinating patterns in human creativity.",
        "Economic theories help us understand market dynamics and consumer behavior.",
        "Educational systems worldwide are adapting to new technological advances."
    ]
    
    ai_texts = [
        "Artificial intelligence systems demonstrate computational capabilities through algorithmic processing.",
        "Environmental sustainability requires systematic approaches to resource optimization frameworks.",
        "Literary analysis reveals structural patterns in narrative construction methodologies.",
        "Economic modeling incorporates statistical analysis of market variable interactions.",
        "Educational frameworks integrate technological solutions for enhanced learning outcomes."
    ]
    
    # Generate samples
    texts = []
    labels = []
    
    for i in range(size):
        if i % 2 == 0:
            texts.append(human_texts[i % len(human_texts)])
            labels.append(0)  # Human
        else:
            texts.append(ai_texts[i % len(ai_texts)])
            labels.append(1)  # AI
    
    return texts, labels

def run_quick_benchmark(models_to_test: Optional[List[str]] = None) -> List[BenchmarkMetrics]:
    """Run a quick benchmark with sample data"""
    # Create sample dataset
    texts, labels = create_sample_dataset(200)
    
    # Initialize benchmark runner
    runner = BenchmarkRunner()
    
    # Run benchmark
    results = runner.run_benchmark_suite(texts, labels, models_to_test)
    
    # Print results
    if results:
        print(runner.generate_comparison_report(results))
        runner.save_results(results)
    
    # Cleanup
    runner.cleanup()
    
    return results

if __name__ == "__main__":
    # Example usage
    print("Running quick benchmark...")
    results = run_quick_benchmark(["bert_base", "distilbert"])
    print(f"Benchmark completed with {len(results)} models.")
