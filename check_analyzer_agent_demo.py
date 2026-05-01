import sys
sys.path.insert(0, '.')
from agents.code_analyzer_agent import code_analyzer_node

state = {
    'code_file_path': 'sample_buggy_code.py',
    'code_content': open('sample_code/sample_buggy_code.py').read(),
    'language': 'python',
    'messages': [], 'errors': [], 'session_id': 'test',
    'start_time': '', 'current_agent': '', 'task_plan': '',
    'assigned_agents': [], 'code_analysis_raw': {},
    'code_quality_score': 0.0, 'code_issues': [],
    'code_analysis_summary': '', 'security_vulnerabilities': [],
    'security_score': 0.0, 'security_risk_level': '',
    'security_summary': '', 'final_report': '', 'report_file_path': '',
}

result = code_analyzer_node(state)
print('Score:', result['code_quality_score'])
print('Issues:', len(result['code_issues']))
print('Summary:', result['code_analysis_summary'])
