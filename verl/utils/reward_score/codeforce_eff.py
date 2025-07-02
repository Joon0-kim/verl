import re
import subprocess
import tempfile
import os


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


def accuracy_reward_code_code_open_r1(content, extra_info):
    test_cases = extra_info["testcases"]
    single_timeout = 2.0
    code_snippet = extract_python_code(content)

    if not code_snippet or not test_cases:
        return 0.0

    def evaluate_code(code, test_cases):
        passed = 0
        total = len(test_cases)
        
        # Create a temporary file to avoid repeated subprocess overhead
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name
        
        try:
            for test_case in test_cases:
                try: 
                    process = subprocess.run(
                        ["python3", temp_file_path],
                        input=test_case["input"],
                        text=True,
                        capture_output=True,
                        timeout=single_timeout 
                    )
        
                    if process.returncode != 0:  # 실행 오류
                        continue  # 다음 테스트케이스로 진행
        
                    output = process.stdout.strip()
                    expected_output = test_case["output"].strip()
        
                    # 전체 출력 비교 (더 효율적)
                    if output == expected_output:
                        passed += 1
        
                except subprocess.TimeoutExpired:
                    # 현재 테스트 케이스가 시간 초과한 경우
                    continue  # 다음 테스트케이스로 진행
                except Exception:
                    # 기타 예외 처리
                    continue
        
        finally:
            # 임시 파일 정리
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass  # 파일이 이미 삭제되었거나 삭제할 수 없는 경우
        
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