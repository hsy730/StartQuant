"""Test genetic mining API"""
import requests
import time
import json

url = 'http://localhost:8000/api/mining/genetic'
payload = {
    "stock_pool_id": "sz50",
    "base_factors": ["alpha084"],
    "start_date": "2025-06-20",
    "end_date": "2026-06-20",
    "population_size": 50,
    "n_generations": 10,
    "cx_prob": 0.7,
    "mut_prob": 0.2,
    "elite_size": 5,
    "fitness_objective": "ic_mean",
    "ic_threshold": 0.03,
    "parsimony_coeff": 0.001,
    "diversity_penalty_coeff": 0.1,
    "cv_folds": 0,
    "use_extended_primitives": True,
    "max_tree_depth": 10,
    "use_nsga2": True,
    "algorithm": "genetic",
    "freq": "D"
}

print(f"Sending POST to {url}...")
t0 = time.time()
try:
    resp = requests.post(url, json=payload, timeout=600)
    elapsed = time.time() - t0
    print(f"Status: {resp.status_code}, Elapsed: {elapsed:.1f}s")
    data = resp.json()
    if isinstance(data, dict):
        print(f"Task ID: {data.get('task_id', 'N/A')}")
        print(f"Status: {data.get('status', 'N/A')}")
        factors = data.get("discovered_factors", [])
        print(f"Discovered factors: {len(factors)}")
        if factors:
            for f in factors[:5]:
                expr = f.get("expression", "?")[:60]
                score = f.get("overall_score", "?")
                passed = f.get("overall_passed", "?")
                print(f"  [{passed}] score={score}  {expr}")
        else:
            # Print more of response for debugging
            print(f"Response keys: {list(data.keys())}")
            print(json.dumps(data, ensure_ascii=False)[:500])
    else:
        print(str(data)[:500])
except Exception as e:
    elapsed = time.time() - t0
    print(f"Error after {elapsed:.1f}s: {e}")
