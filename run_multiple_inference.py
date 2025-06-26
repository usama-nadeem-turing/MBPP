import subprocess
import time
import logging
import os
import json
from datetime import datetime
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultipleInferenceRunner:
    def __init__(self):
        """
        Initialize the multiple inference runner.
        """
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.master_results_dir = f"multiple_inference_{self.timestamp}"
        os.makedirs(self.master_results_dir, exist_ok=True)
        logger.info(f"Created master results directory: {self.master_results_dir}")
        
        # Define the 9 different configurations
        self.configurations = [
            {
                "name": "demo_test_low_temp",
                "args": ["--demo", "--split", "test", "--temperature", "0.1"],
                "description": "Demo mode, test split, low temperature (0.1)"
            },
            {
                "name": "demo_test_high_temp",
                "args": ["--demo", "--split", "test", "--temperature", "0.8"],
                "description": "Demo mode, test split, high temperature (0.8)"
            },
            {
                "name": "demo_train_low_temp",
                "args": ["--demo", "--split", "train", "--temperature", "0.1"],
                "description": "Demo mode, train split, low temperature (0.1)"
            },
            {
                "name": "demo_train_high_temp",
                "args": ["--demo", "--split", "train", "--temperature", "0.8"],
                "description": "Demo mode, train split, high temperature (0.8)"
            },
            {
                "name": "demo_validation_low_temp",
                "args": ["--demo", "--split", "validation", "--temperature", "0.1"],
                "description": "Demo mode, validation split, low temperature (0.1)"
            },
            {
                "name": "demo_validation_high_temp",
                "args": ["--demo", "--split", "validation", "--temperature", "0.8"],
                "description": "Demo mode, validation split, high temperature (0.8)"
            },
            {
                "name": "full_test_medium_temp",
                "args": ["--split", "test", "--max-problems", "10", "--temperature", "0.5"],
                "description": "Full mode, test split, 10 problems, medium temperature (0.5)"
            },
            {
                "name": "full_train_medium_temp",
                "args": ["--split", "train", "--max-problems", "10", "--temperature", "0.5"],
                "description": "Full mode, train split, 10 problems, medium temperature (0.5)"
            },
            {
                "name": "full_validation_medium_temp",
                "args": ["--split", "validation", "--max-problems", "10", "--temperature", "0.5"],
                "description": "Full mode, validation split, 10 problems, medium temperature (0.5)"
            }
        ]
        
        self.results = []
    
    def run_single_inference(self, config: Dict[str, Any], run_number: int) -> Dict[str, Any]:
        """
        Run a single inference with the given configuration.
        
        Args:
            config: Configuration dictionary
            run_number: Current run number (1-9)
            
        Returns:
            Result dictionary with run information
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"RUN {run_number}/9: {config['name']}")
        logger.info(f"Description: {config['description']}")
        logger.info(f"Args: {' '.join(config['args'])}")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Run the inference script
            cmd = ["python", "inference.py"] + config['args']
            logger.info(f"Executing: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout per run
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Check if the run was successful
            success = result.returncode == 0
            
            run_result = {
                "run_number": run_number,
                "config_name": config['name'],
                "description": config['description'],
                "args": config['args'],
                "success": success,
                "duration_seconds": duration,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timestamp": datetime.now().isoformat()
            }
            
            if success:
                logger.info(f"✅ RUN {run_number} COMPLETED SUCCESSFULLY in {duration:.2f} seconds")
            else:
                logger.error(f"❌ RUN {run_number} FAILED after {duration:.2f} seconds")
                logger.error(f"Return code: {result.returncode}")
                if result.stderr:
                    logger.error(f"Error output: {result.stderr}")
            
            return run_result
            
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ RUN {run_number} TIMED OUT after 30 minutes")
            return {
                "run_number": run_number,
                "config_name": config['name'],
                "description": config['description'],
                "args": config['args'],
                "success": False,
                "duration_seconds": 1800,
                "return_code": -1,
                "stdout": "",
                "stderr": "Timeout after 30 minutes",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"💥 RUN {run_number} CRASHED: {e}")
            return {
                "run_number": run_number,
                "config_name": config['name'],
                "description": config['description'],
                "args": config['args'],
                "success": False,
                "duration_seconds": time.time() - start_time,
                "return_code": -1,
                "stdout": "",
                "stderr": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def run_all_inferences(self) -> List[Dict[str, Any]]:
        """
        Run all 9 inference configurations.
        
        Returns:
            List of results from all runs
        """
        logger.info(f"Starting multiple inference runs...")
        logger.info(f"Total configurations: {len(self.configurations)}")
        logger.info(f"Master results directory: {self.master_results_dir}")
        
        all_results = []
        successful_runs = 0
        
        for i, config in enumerate(self.configurations, 1):
            # Add delay between runs to avoid overwhelming the system
            if i > 1:
                logger.info("Waiting 10 seconds before next run...")
                time.sleep(10)
            
            result = self.run_single_inference(config, i)
            all_results.append(result)
            
            if result['success']:
                successful_runs += 1
            
            # Save progress after each run
            self.save_progress(all_results, i)
        
        # Final summary
        logger.info(f"\n{'='*60}")
        logger.info("MULTIPLE INFERENCE COMPLETED")
        logger.info(f"{'='*60}")
        logger.info(f"Total runs: {len(all_results)}")
        logger.info(f"Successful runs: {successful_runs}")
        logger.info(f"Failed runs: {len(all_results) - successful_runs}")
        logger.info(f"Success rate: {successful_runs/len(all_results)*100:.1f}%")
        
        return all_results
    
    def save_progress(self, results: List[Dict[str, Any]], current_run: int):
        """
        Save progress after each run.
        
        Args:
            results: List of results so far
            current_run: Current run number
        """
        progress_file = os.path.join(self.master_results_dir, "progress.json")
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "current_run": current_run,
                "total_runs": len(self.configurations),
                "results": results
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Progress saved: {current_run}/{len(self.configurations)} runs completed")
    
    def save_final_results(self, results: List[Dict[str, Any]]):
        """
        Save final results and summary.
        
        Args:
            results: List of all results
        """
        # Save detailed results
        results_file = os.path.join(self.master_results_dir, "all_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Create summary
        successful_runs = [r for r in results if r['success']]
        failed_runs = [r for r in results if not r['success']]
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_runs": len(results),
            "successful_runs": len(successful_runs),
            "failed_runs": len(failed_runs),
            "success_rate": len(successful_runs) / len(results) if results else 0,
            "average_duration": sum(r['duration_seconds'] for r in results) / len(results) if results else 0,
            "total_duration": sum(r['duration_seconds'] for r in results),
            "configurations": [
                {
                    "name": r['config_name'],
                    "description": r['description'],
                    "success": r['success'],
                    "duration": r['duration_seconds']
                }
                for r in results
            ]
        }
        
        # Save summary
        summary_file = os.path.join(self.master_results_dir, "summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Save human-readable summary
        summary_txt_file = os.path.join(self.master_results_dir, "summary.txt")
        with open(summary_txt_file, 'w', encoding='utf-8') as f:
            f.write("MULTIPLE INFERENCE RESULTS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total runs: {len(results)}\n")
            f.write(f"Successful runs: {len(successful_runs)}\n")
            f.write(f"Failed runs: {len(failed_runs)}\n")
            f.write(f"Success rate: {summary['success_rate']*100:.1f}%\n")
            f.write(f"Average duration: {summary['average_duration']:.2f} seconds\n")
            f.write(f"Total duration: {summary['total_duration']:.2f} seconds\n\n")
            
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 30 + "\n")
            for i, result in enumerate(results, 1):
                status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
                f.write(f"{i}. {result['config_name']}: {status} ({result['duration_seconds']:.2f}s)\n")
                f.write(f"   {result['description']}\n")
                if not result['success'] and result['stderr']:
                    f.write(f"   Error: {result['stderr'][:200]}...\n")
                f.write("\n")
        
        logger.info(f"Final results saved to: {self.master_results_dir}")
        logger.info(f"Summary: {summary_file}")
        logger.info(f"Detailed results: {results_file}")

def main():
    """
    Main function to run multiple inference experiments.
    """
    logger.info("Starting Multiple Inference Runner")
    logger.info("This will run inference.py 9 times with different configurations")
    
    try:
        # Check if inference.py exists
        if not os.path.exists("inference.py"):
            logger.error("inference.py not found in current directory")
            return
        
        # Initialize runner
        runner = MultipleInferenceRunner()
        
        # Run all inferences
        results = runner.run_all_inferences()
        
        # Save final results
        runner.save_final_results(results)
        
        logger.info("Multiple inference experiment completed!")
        
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
    except Exception as e:
        logger.error(f"Error in multiple inference: {e}")
        raise

if __name__ == "__main__":
    main() 