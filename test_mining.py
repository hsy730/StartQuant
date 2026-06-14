import requests
import json
import time

resp = requests.post('http://localhost:8000/api/mining/start', json={
    'stock_codes': ['600036'],
    'start_date': '2024-01-01',
    'end_date': '2024-06-01',
    'base_factors': ['RSI(close, 14)', 'SMA(close, 20)'],
    'algorithm': 'genetic',
    'n_generations': 2,
    'population_size': 10,
})
print(f'Status: {resp.status_code}')
data = resp.json()
task_id = data.get('task_id', '')
print(f'Task ID: {task_id}')
print(f'Response: {json.dumps(data, indent=2, ensure_ascii=False)[:500]}')

if task_id:
    for i in range(60):
        time.sleep(5)
        status_resp = requests.get(f'http://localhost:8000/api/mining/status/{task_id}')
        status = status_resp.json()
        s = status.get('status', '')
        p = status.get('progress', 0)
        print(f'[{i*5}s] Status: {s}, Progress: {p}%')
        if s in ('completed', 'failed', 'cancelled'):
            if s == 'failed':
                print(f'Error: {status.get("error")}')
            elif s == 'completed':
                result = status.get('result', {})
                factors = result.get('factors', [])
                print(f'Discovered {len(factors)} factors')
            break
