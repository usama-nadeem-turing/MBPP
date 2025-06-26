import subprocess
import sys
import os
import time
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_inference(run_number):
    """
    Run inference.py with the same configuration.
    
    Args:
        run_number: Current run number (1-9)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Build the command with the same configuration for all runs
        cmd = [
            sys.executable, "inference.py",
            "--demo",  # Demo mode
            "--split", "test",  # Test split
            "--temperature", "1.0",  # Medium temperature
            "--max-tokens", "512"
        ]
        
        # Log the command being executed
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            logger.info(f"✅ Run {run_number}/9 completed successfully")
            if result.stdout:
                logger.debug(f"Output: {result.stdout}")
            return True
        else:
            logger.error(f"❌ Run {run_number}/9 failed with return code {result.returncode}")
            if result.stderr:
                logger.error(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ Run {run_number}/9 timed out after 1 hour")
        return False
    except Exception as e:
        logger.error(f"💥 Run {run_number}/9 failed with exception: {e}")
        return False

def main():
    """
    Main function to run inference.py 9 times with the same configuration.
    """
    # Configuration used for all runs
    config = {
        'demo': True,
        'split': 'test',
        'temperature': 1.0,
        'max_tokens': 512
    }
    
    # Create a summary directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = f"multiple_inference_run_{timestamp}"
    os.makedirs(summary_dir, exist_ok=True)
    
    logger.info(f"🚀 Starting multiple inference run - 9 times with same configuration")
    logger.info(f"📁 Summary directory: {summary_dir}")
    logger.info(f"⚙️  Configuration: Demo mode, Test split, Temperature=1.0, Max tokens=512")
    
    # Track results
    results = []
    successful_runs = 0
    failed_runs = 0
    
    # Run inference 9 times
    for run_number in range(1, 10):
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 Running inference {run_number}/9")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        success = run_inference(run_number)
        end_time = time.time()
        
        duration = end_time - start_time
        
        result = {
            'run_number': run_number,
            'success': success,
            'duration_seconds': duration,
            'timestamp': datetime.now().isoformat()
        }
        
        results.append(result)
        
        if success:
            successful_runs += 1
            logger.info(f"✅ Run {run_number} completed in {duration:.2f} seconds")
        else:
            failed_runs += 1
            logger.error(f"❌ Run {run_number} failed after {duration:.2f} seconds")
        
        # Add a small delay between runs
        if run_number < 9:
            logger.info("⏳ Waiting 5 seconds before next run...")
            time.sleep(5)
    
    # Save summary results
    summary_file = os.path.join(summary_dir, "run_summary.json")
    summary_data = {
        'timestamp': timestamp,
        'total_runs': 9,
        'successful_runs': successful_runs,
        'failed_runs': failed_runs,
        'success_rate': successful_runs / 9,
        'configuration': config,
        'results': results
    }
    
    try:
        import json
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        logger.info(f"📊 Summary saved to {summary_file}")
    except Exception as e:
        logger.error(f"Error saving summary: {e}")
    
    # Print final summary
    logger.info(f"\n{'='*60}")
    logger.info("🎯 FINAL SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total runs: 9")
    logger.info(f"Successful runs: {successful_runs}")
    logger.info(f"Failed runs: {failed_runs}")
    logger.info(f"Success rate: {successful_runs/9:.2%}")
    
    # Print individual results
    logger.info(f"\n📋 Individual Results:")
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        duration = f"{result['duration_seconds']:.2f}s"
        logger.info(f"Run {result['run_number']:2d}: {status} ({duration})")
    
    logger.info(f"\n🏁 Multiple inference run completed!")
    logger.info(f"📁 Check the results in timestamped folders for each run")

if __name__ == "__main__":
    main() 