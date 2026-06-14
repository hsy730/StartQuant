"""临时测试脚本 — 发送挖掘请求并轮询状态"""
import requests
import json
import time

BASE_URL = "http://localhost:8000/api/mining"

def test_mining():
    # 发送挖掘请求
    payload = {
        "stock_codes": ["600036"],
        "start_date": "2024-01-01",
        "end_date": "2024-06-01",
        "base_factors": ["RSI(close,14)", "SMA(close,20)"],
        "algorithm": "genetic",
        "n_generations": 3,
        "population_size": 20,
        "return_column": "return",
    }

    print("Sending mining request...")
    resp = requests.post(f"{BASE_URL}/genetic", json=payload, timeout=10)
    print(f"Status: {resp.status_code}")
    resp_json = resp.json()
    print(f"Response: {json.dumps(resp_json, indent=2, ensure_ascii=False)[:500]}")

    task_id = resp_json.get("data", {}).get("task_id")
    if not task_id:
        print("No task_id returned!")
        return

    print(f"Task ID: {task_id}")

    # 轮询状态
    for i in range(24):  # 最多等2分钟
        time.sleep(5)
        try:
            status_resp = requests.get(f"{BASE_URL}/status/{task_id}", timeout=10)
            status = status_resp.json()
            s = status.get("status")
            p = status.get("progress")
            e = status.get("error", "")
            print(f"[{i*5}s] Status: {s}, Progress: {p}%, Error: {e}")
            if s in ("completed", "failed", "cancelled"):
                print(f"Full result: {json.dumps(status, indent=2, ensure_ascii=False)[:3000]}")
                break
        except Exception as ex:
            print(f"[{i*5}s] Error polling: {ex}")

if __name__ == "__main__":
    test_mining()
