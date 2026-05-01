import sys
sys.path.insert(0, '.')
from agents.security_scanner_agent import security_scanner_node

state = {
    'code_file_path': 'test.py',
    'code_content': '''
password = "hardcoded123"
import pickle
def run(data): return eval(data)
''',
    'language': 'python',
    'messages': [], 'errors': [], 'session_id': 'test',
    'start_time': '', 'current_agent': '', 'task_plan': '',
    'assigned_agents': [], 'code_analysis_raw': {},
    'code_quality_score': 0.0, 'code_issues': [],
    'code_analysis_summary': '', 'security_vulnerabilities': [],
    'security_score': 0.0, 'security_risk_level': '',
    'security_summary': '', 'final_report': '', 'report_file_path': '',
}

result = security_scanner_node(state)
print('Score:', result['security_score'])
print('Risk:', result['security_risk_level'])
print('Summary:', result['security_summary'])
