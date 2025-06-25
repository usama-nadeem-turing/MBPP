import requests
import json
import time
import os
import argparse
from typing import List, Dict, Any
from datasets import load_dataset
import logging

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
            
            if split not in dataset:
                logger.warning(f"Split '{split}' not found. Available splits: {list(dataset.keys())}")
                split = list(dataset.keys())[0]  # Use first available split
                
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
                        demo_mode: bool = False) -> List[Dict[str, Any]]:
        """
        Process all problems and get model responses.
        
        Args:
            problems: List of problem dictionaries
            max_problems: Maximum number of problems to process (None for all)
            save_results: Whether to save results to file
            demo_mode: If True, process only 3-4 problems for demo
            
        Returns:
            List of results with problems and model responses
        """
        if demo_mode:
            max_problems = 4
            logger.info("Running in DEMO mode - processing 4 problems only")
        
        # Debug: Check the type and structure of problems
        logger.debug(f"Problems type: {type(problems)}")
        logger.debug(f"Problems length: {len(problems)}")
        if len(problems) > 0:
            logger.debug(f"First problem type: {type(problems[0])}")
            logger.debug(f"First problem: {problems[0]}")
        
        # Apply max_problems limit by slicing the problems list
        if max_problems:
            problems = problems[:max_problems]
            logger.info(f"Limited to {len(problems)} problems")
            logger.debug(f"After slicing - First problem type: {type(problems[0]) if problems else 'No problems'}")
            
        results = []
        
        for i, problem in enumerate(problems):
            # Debug: Check if problem is a dictionary
            if not isinstance(problem, dict):
                logger.error(f"Problem {i} is not a dictionary: {type(problem)} - {problem}")
                continue
                
            logger.info(f"Processing problem {i+1}/{len(problems)}: Task {problem.get('task_id', 'unknown')}")
            
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
                    filename = "mbpp_demo_results.json"
                else:
                    filename = "mbpp_results_final.json"
                self.save_results(results, filename)
                logger.info(f"Saved incremental results after task {i+1} to {filename}")
            
            # Save individual result
            self.save_individual_result(result, demo_mode)
            
            # Add delay to avoid overwhelming the server
            time.sleep(0.5)
            
        logger.info(f"Completed processing {len(results)} problems")
        return results
    
    def save_results(self, results: List[Dict[str, Any]], filename: str):
        """
        Save results to a JSON file.
        
        Args:
            results: List of result dictionaries
            filename: Output filename
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def save_individual_result(self, result: Dict[str, Any], demo_mode: bool = False):
        """
        Save individual task result to a separate file.
        
        Args:
            result: Single result dictionary
            demo_mode: Whether running in demo mode
        """
        try:
            mbpp_id = result.get('mbpp_id', 'unknown')
            if demo_mode:
                filename = f"mbpp_demo_task_{mbpp_id}.json"
            else:
                filename = f"mbpp_task_{mbpp_id}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.debug(f"Individual result saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving individual result: {e}")
    
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
                       help='Dataset split to use (train, validation, test)')
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
        # Load MBPP dataset
        problems = inference.load_mbpp_dataset(split=args.split)
        
        # Inspect dataset structure
        inference.inspect_dataset_structure(problems)
        
        # Process problems
        if args.demo:
            logger.info("Starting DEMO mode - processing 4 problems...")
            results = inference.process_problems(
                problems, 
                save_results=True,
                demo_mode=True
            )
        else:
            logger.info("Starting to process problems...")
            results = inference.process_problems(
                problems, 
                max_problems=args.max_problems, 
                save_results=True,
                demo_mode=False
            )
        
        # Analyze results
        analysis = inference.analyze_results(results)
        
        # Print analysis
        logger.info("=== ANALYSIS RESULTS ===")
        for key, value in analysis.items():
            logger.info(f"{key}: {value}")
        
        # Print results as examples
        logger.info("\n=== SAMPLE RESULTS ===")
        for i, result in enumerate(results[:3]):
            logger.info(f"\nProblem {i+1} (MBPP ID: {result['mbpp_id']}):")
            logger.info(f"Prompt: {result['prompt'][:200]}...")
            
            if "error" in result["model_response"]:
                logger.info(f"Error: {result['model_response']['error']}")
            else:
                response_content = result["model_response"].get("choices", [{}])[0].get("message", {}).get("content", "")
                logger.info(f"Response: {response_content[:200]}...")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
