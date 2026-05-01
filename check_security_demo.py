from tools.security_scan_tool import check_security

code = open('sample_code/sample_buggy_code.py').read()
result = check_security(code, file_path='sample_buggy_code.py', run_bandit=False)

print('Score:', result['security_score'])
print('Risk:', result['risk_level'])
print('Total findings:', result['vulnerability_counts']['TOTAL'])
print()
for v in result['vulnerabilities']:
    print(f"  [{v['severity']}] Line {v['line']} - {v['type']}")
    print(f"    CWE: {v['cwe']}")
    print(f"    Fix: {v['fix']}")
    print()
