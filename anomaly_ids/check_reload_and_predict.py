import requests, json

r = requests.post("http://localhost:8000/model/reload")
print("Reload:", r.json())

with open("test_samples_unsw.json") as f:
    cases = json.load(f)["test_cases"]

print()
for sample in cases[:5]:
    r = requests.post("http://localhost:8000/predict", json=sample["data"])
    if r.status_code == 200:
        res = r.json()
        got = "intrusion" if res["is_intrusion"] else "normal"
        match = "OK" if got == sample["expected"] else "WRONG"
        print(f"[{match}] expected={sample['expected']:9} got={got:9} prob={res['intrusion_probability']:.3f} alert={res['alert_level']}")
    else:
        print(f"ERROR {r.status_code}: {r.text[:200]}")
