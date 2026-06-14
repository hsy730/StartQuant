import requests, json

task_id = "8b724d9c-2709-4c42-8802-818f3ec74748"
r = requests.get(f"http://localhost:8000/api/mining/status/{task_id}", timeout=5)
d = r.json().get("data", {})
result = d.get("result", {})
print(json.dumps(result, indent=2, ensure_ascii=False, default=str)[:3000])
