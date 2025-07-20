#!/usr/bin/env python3
"""
Run Baseline Benchmark Script
Executes comprehensive baseline evaluation of AI text detection models
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.core.evaluator import BenchmarkRunner, create_sample_dataset
from benchmarks.core.utils import setup_logging, load_dataset
import logging

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Run AI Text Detection Baseline Benchmark')
    parser.add_argument('--config', type=str, 
                       default='benchmarks/configs/base_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--models', nargs='+', 
                       default=None,
                       help='Specific models to test (default: all)')
    parser.add_argument('--dataset', type=str,
                       default=None,
                       help='Path to dataset file')
    parser.add_argument('--sample-size', type=int,
                       default=1000,
                       help='Number of samples to use (default: 1000)')
    parser.add_argument('--output-dir', type=str,
                       default='benchmarks/results',
                       help='Output directory for results')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("AI TEXT DETECTION BASELINE BENCHMARK")
    logger.info("="*60)
    
    try:
        # Initialize benchmark runner
        logger.info(f"Loading configuration from: {args.config}")
        runner = BenchmarkRunner(args.config)
        
        # Load dataset
        if args.dataset:
            logger.info(f"Loading dataset from: {args.dataset}")
            texts, labels = load_dataset(args.dataset, args.sample_size)
        else:
            logger.info(f"Creating sample dataset with {args.sample_size} samples")
            texts, labels = create_sample_dataset(args.sample_size)
        
        logger.info(f"Dataset loaded: {len(texts)} samples")
        logger.info(f"Distribution - Human: {labels.count(0)}, AI: {labels.count(1)}")
        
        # Run benchmark suite
        logger.info("Starting benchmark evaluation...")
        results = runner.run_benchmark_suite(texts, labels, args.models)
        
        if not results:
            logger.error("No models were successfully evaluated")
            return 1
        
        # Save results
        logger.info("Saving benchmark results...")
        results_path = runner.save_results(results)
        
        # Generate and display report
        logger.info("Generating comparison report...")
        report = runner.generate_comparison_report(results)
        print("\n" + report)
        
        # Export configurations for optimization
        logger.info("Exporting optimization configurations...")
        for result in results:
            try:
                config_path = runner.export_for_optimization(result.model_name)
                logger.info(f"Exported config for {result.model_name}: {config_path}")
            except Exception as e:
                logger.warning(f"Failed to export config for {result.model_name}: {e}")
        
        # Cleanup
        runner.cleanup()
        
        logger.info("="*60)
        logger.info("BASELINE BENCHMARK COMPLETED SUCCESSFULLY")
        logger.info(f"Results saved to: {results_path}")
        logger.info("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Benchmark failed with error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
