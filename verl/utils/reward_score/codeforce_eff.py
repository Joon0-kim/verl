import re
import subprocess
import tempfile
import os
import uuid
import threading
from contextlib import contextmanager


def extract_python_code(text):
    """Extract Python code blocks from text, prioritizing python-marked blocks."""
    pattern_strict = r'```python(.*?)```'
    matches_strict = re.findall(pattern_strict, text, re.DOTALL)
    code_blocks = [match.strip() for match in matches_strict]
    
    if not code_blocks:
        pattern = r'```(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        code_blocks = [match.strip() for match in matches]
    
    # Return the longest code block (most likely to be the complete solution)
    return sorted(code_blocks, key=len)[-1] if code_blocks else None


@contextmanager
def create_temp_code_file(code):
    """Thread-safe temporary file creation with automatic cleanup."""
    # Create unique filename to avoid conflicts
    unique_id = str(uuid.uuid4())
    temp_file_path = os.path.join(tempfile.gettempdir(), f"code_exec_{unique_id}.py")
    
    try:
        # Write code to temporary file
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            f.write(code)
        yield temp_file_path
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        except OSError:
            pass  # File might already be deleted


def evaluate_single_test_case(temp_file_path, test_case, timeout):
    """Evaluate a single test case with proper error handling."""
    try:
        process = subprocess.run(
            ["python3", temp_file_path],
            input=test_case["input"],
            text=True,
            capture_output=True,
            timeout=timeout
        )
        
        if process.returncode != 0:
            return False
        
        output = process.stdout.strip()
        expected_output = test_case["output"].strip()
        
        return output == expected_output
        
    except (subprocess.TimeoutExpired, Exception):
        return False


def accuracy_reward_code_code_open_r1(content, extra_info):
    test_cases = extra_info["testcases"]
    single_timeout = 2.0
    code_snippet = extract_python_code(content)

    if not code_snippet or not test_cases:
        return 0.0

    def evaluate_code(code, test_cases):
        passed = 0
        total = len(test_cases)
        
        # Use context manager for thread-safe file handling
        with create_temp_code_file(code) as temp_file_path:
            for test_case in test_cases:
                if evaluate_single_test_case(temp_file_path, test_case, single_timeout):
                    passed += 1
        
        success_rate = (passed / total) if total > 0 else 0.0
        return success_rate

    success_rate = evaluate_code(code_snippet, test_cases)
    
    # 보상 계산
    if success_rate == 1.0:
        reward = 1.0
        print("All Pass")
    elif success_rate > 0:
        reward = success_rate * 0.1
        print("Partial Pass")
    else:
        reward = 0.0
        print("No Pass")
        
    return reward


def compute_score(solution_str, extra_info) -> float:
    score = accuracy_reward_code_code_open_r1(solution_str, extra_info)
    return score