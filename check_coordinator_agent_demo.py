import sys
sys.path.insert(0, '.')

from agents.coordinator_agent import coordinator_node

state = {
    'code_file_path': 'sample_code/sample_buggy_code.py',
    'code_content': '',
    'language': 'python',
    'messages': [],
    'errors': [],
    'session_id': 'test',
    'start_time': '',
    'current_agent': '',
    'task_plan': '',
    'assigned_agents': [],
    'code_analysis_raw': {},
    'code_quality_score': 0.0,
    'code_issues': [],
    'code_analysis_summary': '',
    'security_vulnerabilities': [],
    'security_score': 0.0,
    'security_risk_level': '',
    'security_summary': '',
    'final_report': '',
    'report_file_path': '',
}

result = coordinator_node(state)

print('Current agent:', result.get('current_agent'))
print('Language:', result.get('language'))
print('Assigned agents:', result.get('assigned_agents'))
print('Start time:', result.get('start_time'))
print('Errors:', result.get('errors', []))
print()

print('Task plan:')
print(result.get('task_plan', ''))
print()

analysis = result.get('coordinator_analysis', {})
if analysis:
    print('Coordinator analysis (tool output):')
    print('  File:', analysis.get('file_path'))
    print('  Purpose:', analysis.get('code_purpose'))
    print('  Priority:', analysis.get('priority'))
    print('  Notes:', analysis.get('analysis_notes'))
