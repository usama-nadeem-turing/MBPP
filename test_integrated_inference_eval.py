#!/usr/bin/env python3
"""
Test script for the integrated inference and evaluation functionality.
This script tests the basic functionality without requiring a model server.
"""

import json
import tempfile
import os
from inference_and_eval_xlsx import MBPPInference

def create_mock_results():
    """Create mock results for testing evaluation functionality."""
    mock_results = [
        {
            "mbpp_id": "test_1",
            "problem": {
                "task_id": "test_1",
                "text": "Write a function that returns the sum of two numbers.",
                "test_list": [
                    "assert add_numbers(2, 3) == 5",
                    "assert add_numbers(-1, 1) == 0"
                ]
            },
            "model_response": {
                "choices": [
                    {
                        "message": {
                            "content": "```python\ndef add_numbers(a, b):\n    return a + b\n```"
                        }
                    }
                ]
            }
        },
        {
            "mbpp_id": "test_2",
            "problem": {
                "task_id": "test_2",
                "text": "Write a function that returns the product of two numbers.",
                "test_list": [
                    "assert multiply_numbers(2, 3) == 6",
                    "assert multiply_numbers(0, 5) == 0"
                ]
            },
            "model_response": {
                "choices": [
                    {
                        "message": {
                            "content": "```python\ndef multiply_numbers(a, b):\n    return a * b\n```"
                        }
                    }
                ]
            }
        },
        {
            "mbpp_id": "test_3",
            "problem": {
                "task_id": "test_3",
                "text": "Write a function that returns the square of a number.",
                "test_list": [
                    "assert square_number(4) == 16",
                    "assert square_number(-2) == 4"
                ]
            },
            "model_response": {
                "choices": [
                    {
                        "message": {
                            "content": "```python\ndef square_number(x):\n    return x ** 2\n```"
                        }
                    }
                ]
            }
        }
    ]
    return mock_results

def test_code_extraction():
    """Test code extraction functionality."""
    print("Testing code extraction...")
    
    inference = MBPPInference()
    
    # Test with code blocks
    response_with_blocks = {
        "choices": [
            {
                "message": {
                    "content": "```python\ndef test_function():\n    return True\n```"
                }
            }
        ]
    }
    
    extracted_code = inference.extract_code_from_response(response_with_blocks)
    print(f"Extracted code from blocks: {extracted_code}")
    assert "def test_function():" in extracted_code
    
    # Test with plain text
    response_plain = {
        "choices": [
            {
                "message": {
                    "content": "def plain_function():\n    return 'hello'"
                }
            }
        ]
    }
    
    extracted_code = inference.extract_code_from_response(response_plain)
    print(f"Extracted code from plain text: {extracted_code}")
    assert "def plain_function():" in extracted_code
    
    print("✓ Code extraction tests passed!")

def test_evaluation():
    """Test evaluation functionality with mock results."""
    print("\nTesting evaluation functionality...")
    
    inference = MBPPInference()
    mock_results = create_mock_results()
    
    # Test evaluation of all results
    evaluations = inference.evaluate_all_results(mock_results)
    
    print(f"Evaluated {len(evaluations)} results")
    
    # Check that all evaluations have the expected structure
    for eval_result in evaluations:
        assert 'mbpp_id' in eval_result
        assert 'passed' in eval_result
        assert 'test_outputs' in eval_result
        assert 'test_cases' in eval_result
    
    # Analyze evaluations
    analysis = inference.analyze_evaluations(evaluations)
    
    print(f"Analysis results:")
    print(f"  Total problems: {analysis['total_problems']}")
    print(f"  Passed: {analysis['passed']}")
    print(f"  Failed: {analysis['failed']}")
    print(f"  Pass rate: {analysis['pass_rate']:.2%}")
    
    # All our mock functions should pass
    assert analysis['total_problems'] == 3
    assert analysis['passed'] == 3
    assert analysis['pass_rate'] == 1.0
    
    print("✓ Evaluation tests passed!")

def test_excel_export():
    """Test Excel export functionality."""
    print("\nTesting Excel export...")
    
    inference = MBPPInference()
    mock_results = create_mock_results()
    evaluations = inference.evaluate_all_results(mock_results)
    
    # Test DataFrame conversion
    df = inference.convert_evaluations_to_dataframe(evaluations)
    print(f"Created DataFrame with {len(df)} rows")
    print(f"DataFrame columns: {list(df.columns)}")
    
    # Check DataFrame structure
    expected_columns = ['mbpp_id', 'test_case_statement', 'test_output', 'reason']
    for col in expected_columns:
        assert col in df.columns
    
    # Test Excel save (to temporary file)
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
        excel_path = tmp_file.name
    
    try:
        inference.save_to_excel(df, excel_path)
        print(f"Excel file saved to: {excel_path}")
        
        # Check if file exists and has content
        assert os.path.exists(excel_path)
        assert os.path.getsize(excel_path) > 0
        
        print("✓ Excel export tests passed!")
        
    finally:
        # Clean up
        if os.path.exists(excel_path):
            os.unlink(excel_path)

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Integrated Inference and Evaluation Functionality")
    print("=" * 60)
    
    try:
        test_code_extraction()
        test_evaluation()
        test_excel_export()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed! Integration is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        raise

if __name__ == "__main__":
    main() 