import requests
import json
import time
import os
import argparse
from typing import List, Dict, Any
from datasets import load_dataset
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MBPPInference:
    def __init__(self, model_url: str = "http://localhost:18000/v1/chat/completions", 
                 model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        """
        Initialize the MBPP inference class.
        
        Args:
            model_url: URL of the local model server
            model_name: Name of the model to use
        """
        self.model_url = model_url
        self.model_name = model_name
        self.session = requests.Session()
        
        # Create results directory with timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"results_{self.timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        logger.info(f"Created results directory: {self.results_dir}")
        
    def load_mbpp_dataset(self, split: str = "test") -> List[Dict[str, Any]]:
        """
        Load the MBPP dataset from Hugging Face.
        
        Args:
            split: Dataset split to load ('train', 'validation', 'test')
            
        Returns:
            List of problem dictionaries
        """
        try:
            logger.info(f"Loading MBPP dataset with split: {split}")
            dataset = load_dataset("mbpp")
            
            # Log available splits
            available_splits = list(dataset.keys())
            logger.info(f"Available splits in MBPP dataset: {available_splits}")
            
            if split not in dataset:
                logger.warning(f"Split '{split}' not found. Available splits: {available_splits}")
                split = available_splits[0]  # Use first available split
                logger.info(f"Using fallback split: {split}")
            else:
                logger.info(f"Successfully using requested split: {split}")
                
            problems = dataset[split]
            logger.info(f"Loaded {len(problems)} problems from MBPP {split} split")
            
            # Debug: Check the type of problems
            logger.debug(f"Problems type: {type(problems)}")
            logger.debug(f"Problems is list: {isinstance(problems, list)}")
            
            # Convert to list if it's not already
            if not isinstance(problems, list):
                logger.warning(f"Problems is not a list, converting from {type(problems)}")
                problems = list(problems)
            
            # Inspect the structure of the first problem
            if len(problems) > 0:
                first_problem = problems[0]
                logger.info(f"First problem structure - Keys: {list(first_problem.keys())}")
                logger.info(f"Sample problem: {first_problem}")
                logger.debug(f"First problem type: {type(first_problem)}")
            
            return problems
            
        except Exception as e:
            logger.error(f"Error loading MBPP dataset: {e}")
            raise
    
    def inspect_dataset_structure(self, problems: List[Dict[str, Any]], num_samples: int = 3):
        """
        Inspect the structure of the dataset to understand available fields.
        
        Args:
            problems: List of problem dictionaries
            num_samples: Number of sample problems to inspect
        """
        logger.info("=== DATASET STRUCTURE INSPECTION ===")
        for i in range(min(num_samples, len(problems))):
            problem = problems[i]
            logger.info(f"\nProblem {i+1} (Task ID: {problem.get('task_id', 'N/A')}):")
            logger.info(f"Keys: {list(problem.keys())}")
            for key, value in problem.items():
                if isinstance(value, str) and len(value) > 100:
                    logger.info(f"{key}: {value[:100]}...")
                else:
                    logger.info(f"{key}: {value}")
        logger.info("=== END INSPECTION ===\n")
    
    def create_prompt(self, problem: Dict[str, Any]) -> str:
        """
        Create a prompt for the model based on the problem.
        
        Args:
            problem: Problem dictionary from MBPP dataset
            
        Returns:
            Formatted prompt string
        """
        # Debug: Print available keys to understand the structure
        logger.debug(f"Available keys in problem: {list(problem.keys())}")
        
        # Get test cases if available
        test_cases = ""
        if 'test_list' in problem and problem['test_list']:
            test_cases = "\n\nExpected behavior (test cases):\n"
            for i, test in enumerate(problem['test_list'], 1):
                # Clean up the assert statement to show the expected behavior
                test_clean = test.replace("assert ", "").replace(" == ", " should return ")
                test_cases += f"{i}. {test_clean}\n"
        
        prompt = f"""Please solve the following Python programming problem:

Problem: {problem['text']}

Task ID: {problem['task_id']}{test_cases}

Please provide a complete Python function that solves this problem. Write only the function code without any explanations or comments."""
        
        return prompt
    
    def get_model_response(self, prompt: str, temperature: float = 0.2, 
                          max_tokens: int = 512) -> Dict[str, Any]:
        """
        Get response from the local model server.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Model response dictionary
        """
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = self.session.post(
                self.model_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling model API: {e}")
            return {"error": str(e)}
    
    def process_problems(self, problems: List[Dict[str, Any]], 
                        max_problems: int = None, 
                        save_results: bool = True,
                        demo_mode: bool = False,
                        split_name: str = "") -> List[Dict[str, Any]]:
        """
        Process all problems and get model responses.
        
        Args:
            problems: List of problem dictionaries
            max_problems: Maximum number of problems to process (None for all)
            save_results: Whether to save results to file
            demo_mode: If True, process only 3-4 problems for demo
            split_name: Name of the split being processed
            
        Returns:
            List of results with problems and model responses
        """
        if demo_mode:
            max_problems = 4
            logger.info(f"Running in DEMO mode - processing 4 problems only for split: {split_name}")
        
        # Debug: Check the type and structure of problems
        logger.debug(f"Problems type: {type(problems)}")
        logger.debug(f"Problems length: {len(problems)}")
        if len(problems) > 0:
            logger.debug(f"First problem type: {type(problems[0])}")
            logger.debug(f"First problem: {problems[0]}")
        
        # Apply max_problems limit by slicing the problems list
        if max_problems:
            problems = problems[:max_problems]
            logger.info(f"Limited to {len(problems)} problems for split: {split_name}")
            logger.debug(f"After slicing - First problem type: {type(problems[0]) if problems else 'No problems'}")
            
        results = []
        
        for i, problem in enumerate(problems):
            # Debug: Check if problem is a dictionary
            if not isinstance(problem, dict):
                logger.error(f"Problem {i} is not a dictionary: {type(problem)} - {problem}")
                continue
                
            logger.info(f"Processing problem {i+1}/{len(problems)}: Task {problem.get('task_id', 'unknown')} for split: {split_name}")
            
            # Create prompt
            prompt = self.create_prompt(problem)
            
            # Get model response
            model_response = self.get_model_response(prompt)
            
            # Store result with MBPP ID
            result = {
                "mbpp_id": problem.get('task_id', f'unknown_{i}'),
                "problem": problem,
                "prompt": prompt,
                "model_response": model_response,
                "timestamp": time.time()
            }
            
            results.append(result)
            
            # Save result incrementally after each task
            if save_results:
                if demo_mode:
                    filename = f"mbpp_demo_results_{split_name}.json"
                else:
                    filename = f"mbpp_results_final_{split_name}.json"
                self.save_results(results, filename)
                logger.info(f"Saved incremental results after task {i+1} to {filename}")
            
            # Save individual result
            self.save_individual_result(result, demo_mode)
            
            # Add delay to avoid overwhelming the server
            time.sleep(0.5)
            
        logger.info(f"Completed processing {len(results)} problems for split: {split_name}")
        return results
    
    def save_results(self, results: List[Dict[str, Any]], filename: str):
        """
        Save results to a JSON file in the results directory.
        
        Args:
            results: List of result dictionaries
            filename: Output filename
        """
        try:
            filepath = os.path.join(self.results_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def save_individual_result(self, result: Dict[str, Any], demo_mode: bool = False):
        """
        Save individual task result to a separate file in the results directory.
        
        Args:
            result: Single result dictionary
            demo_mode: Whether running in demo mode
        """
        try:
            mbpp_id = result.get('mbpp_id', 'unknown')
            if demo_mode:
                filename = f"mbpp_demo_task_{mbpp_id}_{self.timestamp}.json"
            else:
                filename = f"mbpp_task_{mbpp_id}_{self.timestamp}.json"
            
            filepath = os.path.join(self.results_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.debug(f"Individual result saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving individual result: {e}")
    
    def save_run_metadata(self, args, problems_count: int, demo_mode: bool = False, split: str = ""):
        """
        Save run metadata and configuration to the results directory.
        
        Args:
            args: Command line arguments
            problems_count: Number of problems to be processed
            demo_mode: Whether running in demo mode
            split: Name of the split being processed
        """
        try:
            metadata = {
                "timestamp": self.timestamp,
                "run_config": {
                    "demo_mode": demo_mode,
                    "split": split,
                    "max_problems": args.max_problems,
                    "model_url": args.model_url,
                    "model_name": args.model_name,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "debug": args.debug
                },
                "dataset_info": {
                    "total_problems_loaded": problems_count,
                    "problems_to_process": min(problems_count, args.max_problems or problems_count) if not demo_mode else 4
                },
                "results_directory": self.results_dir
            }
            
            filepath = os.path.join(self.results_dir, f"run_metadata_{split}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Run metadata saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving run metadata: {e}")
    
    def save_analysis(self, analysis: Dict[str, Any], demo_mode: bool = False, split: str = ""):
        """
        Save analysis results to the results directory.
        
        Args:
            analysis: Analysis dictionary
            demo_mode: Whether running in demo mode
            split: Name of the split being processed
        """
        try:
            if demo_mode:
                filename = f"analysis_demo_{split}.json"
            else:
                filename = f"analysis_final_{split}.json"
            
            filepath = os.path.join(self.results_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            logger.info(f"Analysis saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the results and provide statistics.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Analysis dictionary with statistics
        """
        total_problems = len(results)
        successful_responses = sum(1 for r in results if "error" not in r["model_response"])
        failed_responses = total_problems - successful_responses
        
        # Calculate average response time (if available)
        response_times = []
        for r in results:
            if "model_response" in r and "usage" in r["model_response"]:
                usage = r["model_response"]["usage"]
                if "total_tokens" in usage:
                    response_times.append(usage["total_tokens"])
        
        analysis = {
            "total_problems": total_problems,
            "successful_responses": successful_responses,
            "failed_responses": failed_responses,
            "success_rate": successful_responses / total_problems if total_problems > 0 else 0,
            "average_tokens": sum(response_times) / len(response_times) if response_times else 0,
            "total_tokens": sum(response_times) if response_times else 0,
            "mbpp_ids": [r["mbpp_id"] for r in results]
        }
        
        return analysis

def main():
    """
    Main function to run the MBPP inference.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run MBPP inference with local model server')
    parser.add_argument('--demo', action='store_true', 
                       help='Run in demo mode with only 3-4 problems')
    parser.add_argument('--split', type=str, default='test',
                       help='Dataset split to use (train, validation, test). Use "all" to process all splits.')
    parser.add_argument('--max-problems', type=int, default=None,
                       help='Maximum number of problems to process')
    parser.add_argument('--model-url', type=str, 
                       default='http://localhost:18000/v1/chat/completions',
                       help='URL of the model server')
    parser.add_argument('--model-name', type=str, 
                       default='Qwen/Qwen2.5-1.5B-Instruct',
                       help='Name of the model to use')
    parser.add_argument('--temperature', type=float, default=0.2,
                       help='Sampling temperature')
    parser.add_argument('--max-tokens', type=int, default=512,
                       help='Maximum tokens to generate')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Initialize the inference class
    inference = MBPPInference(
        model_url=args.model_url,
        model_name=args.model_name
    )
    
    try:
        # Load MBPP dataset to get available splits
        logger.info("Loading MBPP dataset to check available splits...")
        temp_dataset = load_dataset("mbpp")
        available_splits = list(temp_dataset.keys())
        logger.info(f"Available splits in MBPP dataset: {available_splits}")
        
        # Determine which splits to process
        if args.split.lower() == 'all':
            splits_to_process = available_splits
            logger.info(f"Processing all splits: {splits_to_process}")
        else:
            if args.split not in available_splits:
                logger.error(f"Split '{args.split}' not found. Available splits: {available_splits}")
                return
            splits_to_process = [args.split]
            logger.info(f"Processing specific split: {args.split}")
        
        all_results = {}
        all_analyses = {}
        
        # Process each split
        for split in splits_to_process:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing split: {split}")
            logger.info(f"{'='*50}")
            
            # Load problems for this split
            problems = inference.load_mbpp_dataset(split=split)
            
            # Inspect dataset structure
            inference.inspect_dataset_structure(problems)
            
            # Save run metadata for this split
            inference.save_run_metadata(args, len(problems), args.demo, split)
            
            # Process problems
            if args.demo:
                logger.info(f"Starting DEMO mode for split {split} - processing 4 problems...")
                results = inference.process_problems(
                    problems, 
                    save_results=True,
                    demo_mode=True,
                    split_name=split
                )
            else:
                logger.info(f"Starting to process problems for split {split}...")
                results = inference.process_problems(
                    problems, 
                    max_problems=args.max_problems, 
                    save_results=True,
                    demo_mode=False,
                    split_name=split
                )
            
            # Analyze results for this split
            analysis = inference.analyze_results(results)
            all_results[split] = results
            all_analyses[split] = analysis
            
            # Save analysis for this split
            inference.save_analysis(analysis, args.demo, split)
            
            # Print analysis for this split
            logger.info(f"\n=== ANALYSIS RESULTS FOR SPLIT: {split} ===")
            for key, value in analysis.items():
                logger.info(f"{key}: {value}")
            
            # Print sample results for this split
            logger.info(f"\n=== SAMPLE RESULTS FOR SPLIT: {split} ===")
            for i, result in enumerate(results[:3]):
                logger.info(f"\nProblem {i+1} (MBPP ID: {result['mbpp_id']}):")
                logger.info(f"Prompt: {result['prompt'][:200]}...")
                
                if "error" in result["model_response"]:
                    logger.info(f"Error: {result['model_response']['error']}")
                else:
                    response_content = result["model_response"].get("choices", [{}])[0].get("message", {}).get("content", "")
                    logger.info(f"Response: {response_content[:200]}...")
        
        # Print overall summary if processing multiple splits
        if len(splits_to_process) > 1:
            logger.info(f"\n{'='*50}")
            logger.info("OVERALL SUMMARY")
            logger.info(f"{'='*50}")
            total_problems = sum(len(results) for results in all_results.values())
            total_successful = sum(analysis['successful_responses'] for analysis in all_analyses.values())
            overall_success_rate = total_successful / total_problems if total_problems > 0 else 0
            
            logger.info(f"Total problems processed: {total_problems}")
            logger.info(f"Total successful responses: {total_successful}")
            logger.info(f"Overall success rate: {overall_success_rate:.2%}")
            
            for split in splits_to_process:
                analysis = all_analyses[split]
                logger.info(f"{split}: {analysis['successful_responses']}/{analysis['total_problems']} successful ({analysis['success_rate']:.2%})")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
