import sys
sys.path.insert(0, '.')
from tools.code_analysis_tool import analyze_code_quality

code = open('sample_code/sample_buggy_code.py').read()
result = analyze_code_quality(code, file_path='sample_buggy_code.py', language='python')

print('Score:', result['quality_score'])
print('Total issues:', result['issue_counts']['TOTAL'])
print(f"  HIGH: {result['issue_counts']['HIGH']}  MEDIUM: {result['issue_counts']['MEDIUM']}  LOW: {result['issue_counts']['LOW']}")
print()

for issue in result['issues']:
    print(f"  [{issue['severity']}] Line {issue['line']} - {issue['type']}")
    print(f"    Message: {issue['message']}")
    if issue.get('fix'):
        print(f"    Fix: {issue['fix']}")
    print()
