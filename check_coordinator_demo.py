import sys
sys.path.insert(0, '.')

from tools.coordinator_tool import analyze_file_for_routing

code = open('sample_code/sample_buggy_code.py', encoding='utf-8').read()
result = analyze_file_for_routing(
    file_path='sample_code/sample_buggy_code.py',
    code_content=code,
)

print('Language:', result['language'])
print('Confidence:', result['confidence'])
print('Priority:', result['priority'])
print('Purpose:', result['code_purpose'])
print('Agents:', ', '.join(result['routing_agents']))
print()

metrics = result['metrics']
print('Metrics:')
print('  Total lines:', metrics['total_lines'])
print('  Code lines:', metrics['code_lines'])
print('  Comment lines:', metrics['comment_lines'])
print('  Functions:', metrics['function_count'])
print('  Classes:', metrics['class_count'])
print('  Imports:', metrics['import_count'])
print('  Complexity:', metrics['estimated_complexity'])
print()

print('Complexity assessment:')
print(result['complexity_assessment'])
print()
print('Analysis notes:')
print(result['analysis_notes'])
