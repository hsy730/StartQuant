import requests, json

task_id = "8b724d9c-2709-4c42-8802-818f3ec74748"
r = requests.get(f"http://localhost:8000/api/mining/status/{task_id}", timeout=5)
d = r.json().get("data", {})
print("Status:", d.get("status"))
print("Progress:", d.get("progress"))
print("Best fitness:", d.get("best_fitness"))

result = d.get("result", {})
factors = result.get("factors", [])
print(f"Factors found: {len(factors)}")
for f in factors[:5]:
    name = f.get("name", "")
    expr = f.get("expression", "")[:60]
    ic = f.get("ic")
    ir = f.get("ir")
    passed = f.get("overall_passed")
    gen_id = f.get("generated_factor_id")
    print(f"  {name}: expr={expr}, ic={ic}, ir={ir}, passed={passed}, db_id={gen_id}")
