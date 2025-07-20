
"""
Results Management Module for AI Text Detection Benchmark

Handles saving, loading, and displaying benchmark results in organized formats.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ResultsManager:
    """
    Manages benchmark results including saving, loading, and visualization.
    """
    
    def __init__(self, project_root: Path, experiment_name: str = "baseline"):
        """
        Initialize ResultsManager.
        
        Args:
            project_root: Path to project root directory
            experiment_name: Name of the experiment (e.g., 'baseline', 'optimization')
        """
        self.project_root = project_root
        self.experiment_name = experiment_name
        
        # Create organized directory structure
        self.results_dir = project_root / "benchmarks" / "results" / experiment_name
        self.csv_dir = self.results_dir / "csv"
        self.plots_dir = self.results_dir / "plots"
        self.reports_dir = self.results_dir / "reports"
        
        # Create directories
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ResultsManager initialized for experiment: {experiment_name}")
    
    def save_results_csv(self, results_df: pd.DataFrame, filename: str = None) -> Path:
        """
        Save results DataFrame to CSV.
        
        Args:
            results_df: DataFrame containing results
            filename: Optional custom filename
            
        Returns:
            Path to saved CSV file
        """
        if filename is None:
            filename = f"{self.experiment_name}_results.csv"
        
        csv_path = self.csv_dir / filename
        results_df.to_csv(csv_path, index=False)
        
        logger.info(f"Results CSV saved to: {csv_path}")
        return csv_path
    
    def save_json_report(self, report_data: Dict[str, Any], filename: str = None) -> Path:
        """
        Save structured report as JSON.
        
        Args:
            report_data: Dictionary containing report data
            filename: Optional custom filename
            
        Returns:
            Path to saved JSON file
        """
        if filename is None:
            filename = f"{self.experiment_name}_report.json"
        
        json_path = self.reports_dir / filename
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"JSON report saved to: {json_path}")
        return json_path
    
    def save_text_summary(self, results_df: pd.DataFrame, best_performers: Dict[str, Any], 
                         filename: str = None) -> Path:
        """
        Save human-readable text summary.
        
        Args:
            results_df: DataFrame containing results
            best_performers: Dictionary with best performing models
            filename: Optional custom filename
            
        Returns:
            Path to saved text file
        """
        if filename is None:
            filename = f"{self.experiment_name}_summary.txt"
        
        summary_path = self.reports_dir / filename
        
        with open(summary_path, 'w') as f:
            f.write(f"AI Text Detection Benchmark - {self.experiment_name.title()} Results\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Models Evaluated: {len(results_df['model_name'].unique())}\n")
            f.write(f"Datasets Used: {len(results_df['dataset'].unique())}\n")
            f.write(f"Total Evaluations: {len(results_df)}\n\n")
            
            # Performance Statistics
            f.write("PERFORMANCE STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Mean Accuracy: {results_df['accuracy'].mean():.3f} ± {results_df['accuracy'].std():.3f}\n")
            f.write(f"Mean Precision: {results_df['precision'].mean():.3f} ± {results_df['precision'].std():.3f}\n")
            f.write(f"Mean Recall: {results_df['recall'].mean():.3f} ± {results_df['recall'].std():.3f}\n")
            f.write(f"Mean F1-Score: {results_df['f1_score'].mean():.3f} ± {results_df['f1_score'].std():.3f}\n")
            f.write(f"Mean ROC-AUC: {results_df['roc_auc'].mean():.3f} ± {results_df['roc_auc'].std():.3f}\n")
            f.write(f"Mean Inference Time: {results_df['inference_time'].mean():.3f}s ± {results_df['inference_time'].std():.3f}s\n\n")
            
            # Best Performers
            f.write("BEST PERFORMERS\n")
            f.write("-" * 40 + "\n")
            for metric, performer in best_performers.items():
                f.write(f"{metric}: {performer['model']} ({performer['score']:.3f})\n")
            f.write("\n")
            
            # Model Rankings
            f.write("MODEL RANKINGS\n")
            f.write("-" * 40 + "\n")
            model_avg = results_df.groupby('model_name').agg({
                'accuracy': 'mean',
                'f1_score': 'mean',
                'roc_auc': 'mean',
                'inference_time': 'mean'
            }).round(3)
            
            # Sort by F1-score
            model_avg_sorted = model_avg.sort_values('f1_score', ascending=False)
            
            f.write("Ranked by F1-Score:\n")
            for i, (model, row) in enumerate(model_avg_sorted.iterrows(), 1):
                f.write(f"{i:2d}. {model:30s} | F1: {row['f1_score']:.3f} | Acc: {row['accuracy']:.3f} | Time: {row['inference_time']:.2f}s\n")
            f.write("\n")
            
            # Dataset Performance
            f.write("DATASET PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            dataset_avg = results_df.groupby('dataset').agg({
                'accuracy': 'mean',
                'f1_score': 'mean',
                'roc_auc': 'mean'
            }).round(3)
            
            for dataset, row in dataset_avg.iterrows():
                f.write(f"{dataset:20s} | F1: {row['f1_score']:.3f} | Acc: {row['accuracy']:.3f} | AUC: {row['roc_auc']:.3f}\n")
        
        logger.info(f"Text summary saved to: {summary_path}")
        return summary_path
    
    def generate_performance_plots(self, results_df: pd.DataFrame) -> List[Path]:
        """
        Generate comprehensive performance visualization plots.
        
        Args:
            results_df: DataFrame containing results
            
        Returns:
            List of paths to generated plot files
        """
        plot_paths = []
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("hugging_face")
        
        # 1. Performance Comparison (Box plots)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy
        sns.boxplot(data=results_df, x='model_name', y='accuracy', ax=axes[0,0])
        axes[0,0].set_title('Model Accuracy Distribution')
        axes[0,0].set_xticklabels(axes[0,0].get_xticklabels(), rotation=45)
        axes[0,0].set_ylabel('Accuracy')
        
        # F1-Score
        sns.boxplot(data=results_df, x='model_name', y='f1_score', ax=axes[0,1])
        axes[0,1].set_title('Model F1-Score Distribution')
        axes[0,1].set_xticklabels(axes[0,1].get_xticklabels(), rotation=45)
        axes[0,1].set_ylabel('F1-Score')
        
        # ROC-AUC
        sns.boxplot(data=results_df, x='model_name', y='roc_auc', ax=axes[1,0])
        axes[1,0].set_title('Model ROC-AUC Distribution')
        axes[1,0].set_xticklabels(axes[1,0].get_xticklabels(), rotation=45)
        axes[1,0].set_ylabel('ROC-AUC')
        
        # Inference Time
        sns.boxplot(data=results_df, x='model_name', y='inference_time', ax=axes[1,1])
        axes[1,1].set_title('Model Inference Time Distribution')
        axes[1,1].set_xticklabels(axes[1,1].get_xticklabels(), rotation=45)
        axes[1,1].set_ylabel('Inference Time (s)')
        
        plt.tight_layout()
        plot_path = self.plots_dir / "performance_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(plot_path)
        
        # 2. Performance vs Efficiency Trade-off
        plt.figure(figsize=(12, 8))
        for model in results_df['model_name'].unique():
            model_data = results_df[results_df['model_name'] == model]
            plt.scatter(model_data['inference_time'], model_data['f1_score'], 
                       label=model, s=100, alpha=0.7)
        
        plt.xlabel('Inference Time (seconds)')
        plt.ylabel('F1-Score')
        plt.title('Performance vs Efficiency Trade-off')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = self.plots_dir / "performance_vs_efficiency.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(plot_path)
        
        # 3. Dataset-Model Heatmap
        if len(results_df['dataset'].unique()) > 1:
            plt.figure(figsize=(12, 8))
            heatmap_data = results_df.pivot_table(
                values='f1_score', 
                index='dataset', 
                columns='model_name', 
                aggfunc='mean'
            )
            
            sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.3f')
            plt.title('F1-Score by Dataset and Model')
            plt.ylabel('Dataset')
            plt.xlabel('Model')
            plt.tight_layout()
            
            plot_path = self.plots_dir / "dataset_model_heatmap.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
        
        # 4. Model Ranking Bar Chart
        plt.figure(figsize=(12, 8))
        model_avg = results_df.groupby('model_name')['f1_score'].mean().sort_values(ascending=True)
        
        bars = plt.barh(range(len(model_avg)), model_avg.values)
        plt.yticks(range(len(model_avg)), model_avg.index)
        plt.xlabel('Average F1-Score')
        plt.title('Model Performance Ranking (F1-Score)')
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{model_avg.iloc[i]:.3f}', va='center')
        
        plt.tight_layout()
        plot_path = self.plots_dir / "model_ranking.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(plot_path)
        
        logger.info(f"Generated {len(plot_paths)} performance plots")
        return plot_paths
    
    def print_results_summary(self, results_df: pd.DataFrame):
        """
        Print comprehensive results summary to console.
        
        Args:
            results_df: DataFrame containing results
        """
        print(f"\n🎯 {self.experiment_name.upper()} RESULTS SUMMARY")
        print("=" * 80)
        
        # Basic stats
        print(f"📊 Models Evaluated: {len(results_df['model_name'].unique())}")
        print(f"📊 Datasets Used: {len(results_df['dataset'].unique())}")
        print(f"📊 Total Evaluations: {len(results_df)}")
        
        # Performance overview
        print(f"\n📈 PERFORMANCE OVERVIEW")
        print("-" * 40)
        print(f"Mean Accuracy: {results_df['accuracy'].mean():.3f} ± {results_df['accuracy'].std():.3f}")
        print(f"Mean F1-Score: {results_df['f1_score'].mean():.3f} ± {results_df['f1_score'].std():.3f}")
        print(f"Mean ROC-AUC: {results_df['roc_auc'].mean():.3f} ± {results_df['roc_auc'].std():.3f}")
        print(f"Mean Inference Time: {results_df['inference_time'].mean():.2f}s ± {results_df['inference_time'].std():.2f}s")
        
        # Best performers
        best_accuracy = results_df.loc[results_df['accuracy'].idxmax()]
        best_f1 = results_df.loc[results_df['f1_score'].idxmax()]
        best_roc_auc = results_df.loc[results_df['roc_auc'].idxmax()]
        fastest_model = results_df.loc[results_df['inference_time'].idxmin()]
        
        print(f"\n🏆 BEST PERFORMERS")
        print("-" * 40)
        print(f"🎯 Highest Accuracy: {best_accuracy['model_name']} ({best_accuracy['accuracy']:.3f})")
        print(f"🎯 Highest F1-Score: {best_f1['model_name']} ({best_f1['f1_score']:.3f})")
        print(f"🎯 Highest ROC-AUC: {best_roc_auc['model_name']} ({best_roc_auc['roc_auc']:.3f})")
        print(f"⚡ Fastest Inference: {fastest_model['model_name']} ({fastest_model['inference_time']:.2f}s)")
        
        # Model rankings
        print(f"\n📊 MODEL RANKINGS (by F1-Score)")
        print("-" * 40)
        model_avg = results_df.groupby('model_name')['f1_score'].mean().sort_values(ascending=False)
        
        for i, (model, f1_score) in enumerate(model_avg.items(), 1):
            model_data = results_df[results_df['model_name'] == model]
            avg_time = model_data['inference_time'].mean()
            print(f"{i:2d}. {model:30s} | F1: {f1_score:.3f} | Time: {avg_time:.2f}s")
    
    def save_all_results(self, results_df: pd.DataFrame, 
                        additional_data: Dict[str, Any] = None) -> Dict[str, Path]:
        """
        Save all results in organized format.
        
        Args:
            results_df: DataFrame containing results
            additional_data: Optional additional data for JSON report
            
        Returns:
            Dictionary with paths to saved files
        """
        saved_files = {}
        
        # Save CSV
        saved_files['csv'] = self.save_results_csv(results_df)
        
        # Calculate best performers
        best_performers = {
            'Highest Accuracy': {
                'model': results_df.loc[results_df['accuracy'].idxmax(), 'model_name'],
                'score': float(results_df['accuracy'].max())
            },
            'Highest F1-Score': {
                'model': results_df.loc[results_df['f1_score'].idxmax(), 'model_name'],
                'score': float(results_df['f1_score'].max())
            },
            'Highest ROC-AUC': {
                'model': results_df.loc[results_df['roc_auc'].idxmax(), 'model_name'],
                'score': float(results_df['roc_auc'].max())
            },
            'Fastest Inference': {
                'model': results_df.loc[results_df['inference_time'].idxmin(), 'model_name'],
                'score': float(results_df['inference_time'].min())
            }
        }
        
        # Prepare JSON report
        report_data = {
            'experiment_info': {
                'name': self.experiment_name,
                'timestamp': datetime.now().isoformat(),
                'models_evaluated': len(results_df['model_name'].unique()),
                'datasets_used': len(results_df['dataset'].unique()),
                'total_evaluations': len(results_df)
            },
            'summary_statistics': {
                'mean_accuracy': float(results_df['accuracy'].mean()),
                'std_accuracy': float(results_df['accuracy'].std()),
                'mean_f1_score': float(results_df['f1_score'].mean()),
                'std_f1_score': float(results_df['f1_score'].std()),
                'mean_roc_auc': float(results_df['roc_auc'].mean()),
                'std_roc_auc': float(results_df['roc_auc'].std()),
                'mean_inference_time': float(results_df['inference_time'].mean()),
                'std_inference_time': float(results_df['inference_time'].std())
            },
            'best_performers': best_performers
        }
        
        # Add additional data if provided
        if additional_data:
            report_data.update(additional_data)
        
        # Save JSON report
        saved_files['json_report'] = self.save_json_report(report_data)
        
        # Save text summary
        saved_files['text_summary'] = self.save_text_summary(results_df, best_performers)
        
        # Generate plots
        plot_paths = self.generate_performance_plots(results_df)
        saved_files['plots'] = plot_paths
        
        # Print summary
        self.print_results_summary(results_df)
        
        # Print save locations
        print(f"\n📁 RESULTS SAVED TO:")
        print("-" * 40)
        print(f"📊 CSV Data: {self.csv_dir}")
        print(f"📈 Plots: {self.plots_dir}")
        print(f"📄 Reports: {self.reports_dir}")
        
        return saved_files
    
    def load_results(self, filename: str = None) -> pd.DataFrame:
        """
        Load results from CSV file.
        
        Args:
            filename: Optional specific filename to load
            
        Returns:
            DataFrame with loaded results
        """
        if filename is None:
            filename = f"{self.experiment_name}_results.csv"
        
        csv_path = self.csv_dir / filename
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Results file not found: {csv_path}")
        
        results_df = pd.read_csv(csv_path)
        logger.info(f"Loaded results from: {csv_path}")
        
        return results_df
    
    def compare_experiments(self, other_experiment: str) -> pd.DataFrame:
        """
        Compare current experiment with another experiment.
        
        Args:
            other_experiment: Name of other experiment to compare with
            
        Returns:
            DataFrame with comparison results
        """
        # Load current results
        current_results = self.load_results()
        current_results['experiment'] = self.experiment_name
        
        # Load other experiment results
        other_manager = ResultsManager(self.project_root, other_experiment)
        other_results = other_manager.load_results()
        other_results['experiment'] = other_experiment
        
        # Combine results
        combined_results = pd.concat([current_results, other_results], ignore_index=True)
        
        return combined_results
