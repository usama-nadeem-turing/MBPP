import re

# Your actual output
example_output = """Please solve the following Python programming problem:
Write a function to find length of the string.

Please provide a complete Python function that solves this problem. Write only the function code without any explanations or comments. 
Use this format to encapsulate the generate code <TAG>
def len_str(s):
    # Your solution here
    
    return len(s)  # Return the length of the input string s

# Test cases
print(len_str("Hello"))   # Expected output: 5
print(len_str("Python"))  # Expected output: 6
print(len_str(""))        # Expected output: 0"""

def extract_code_from_output(output_text):
    # First try to find content between <TAG> and </TAG>
    tag_pattern = r'<TAG>(.*?)</TAG>'
    match = re.search(tag_pattern, output_text, re.DOTALL)
    
    if match:
        code = match.group(1).strip()
    else:
        # If no closing tag, find content after <TAG>
        tag_start = output_text.find('<TAG>')
        if tag_start != -1:
            # Get everything after <TAG>
            code_after_tag = output_text[tag_start + 5:].strip()
            
            # Try to find the end of the function (look for patterns that indicate end)
            # Look for common patterns that indicate the end of the main function
            end_patterns = [
                r'# Test cases',
                r'print\(',
                r'import ',
                r'def [a-zA-Z_][a-zA-Z0-9_]*\(',
                r'<TAG>'
            ]
            
            # Find the earliest occurrence of any end pattern
            earliest_end = len(code_after_tag)
            for pattern in end_patterns:
                match = re.search(pattern, code_after_tag)
                if match and match.start() < earliest_end:
                    earliest_end = match.start()
            
            code = code_after_tag[:earliest_end].strip()
        else:
            # Fallback: try to find the function definition
            code = output_text
    
    # Clean up the code - remove test cases and extra content but keep function code
    lines = code.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip empty lines, test cases, and imports that come after the function
        if (line and 
            not line.startswith('print(') and
            not line.startswith('import ') and
            'Expected output:' not in line and
            not line.startswith('<TAG>') and
            not line.startswith('# Test cases')):
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

print("Original output:")
print(example_output)
print("\n" + "="*50 + "\n")

extracted_code = extract_code_from_output(example_output)
print("Extracted code:")
print(extracted_code) 