import json
import urllib.request
import urllib.error
from pathlib import Path

# Config
BOT_URL = "http://127.0.0.1:8080"
DATASET_DIR = Path("dataset")

def post_json(path, data):
    url = f"{BOT_URL}{path}"
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as f:
        return json.loads(f.read().decode("utf-8"))

def main():
    print("Loading test_pairs.json...")
    with open(DATASET_DIR / "test_pairs.json") as f:
        pairs = json.load(f)["pairs"]

    print("Pushing contexts to local bot...")
    # Load categories
    for path in (DATASET_DIR / "categories").glob("*.json"):
        with open(path) as f:
            cat = json.load(f)
        post_json("/v1/context", {"scope": "category", "context_id": cat["slug"], "version": 1, "payload": cat, "delivered_at": "2026-07-03T00:00:00Z"})

    # Load merchants
    for path in (DATASET_DIR / "merchants").glob("*.json"):
        with open(path) as f:
            m = json.load(f)
        post_json("/v1/context", {"scope": "merchant", "context_id": m["merchant_id"], "version": 1, "payload": m, "delivered_at": "2026-07-03T00:00:00Z"})

    # Load customers
    for path in (DATASET_DIR / "customers").glob("*.json"):
        with open(path) as f:
            c = json.load(f)
        post_json("/v1/context", {"scope": "customer", "context_id": c["customer_id"], "version": 1, "payload": c, "delivered_at": "2026-07-03T00:00:00Z"})

    # Load triggers
    for path in (DATASET_DIR / "triggers").glob("*.json"):
        with open(path) as f:
            t = json.load(f)
        post_json("/v1/context", {"scope": "trigger", "context_id": t["id"], "version": 1, "payload": t, "delivered_at": "2026-07-03T00:00:00Z"})

    print("Generating submission.jsonl...")
    out_lines = []
    for pair in pairs:
        test_id = pair["test_id"]
        trg_id = pair["trigger_id"]
        
        # Trigger tick
        res = post_json("/v1/tick", {"now": "2026-07-03T00:00:00Z", "available_triggers": [trg_id]})
        actions = res.get("actions", [])
        if not actions:
            print(f"Warning: No action generated for {test_id}")
            continue
            
        action = actions[0]
        out_lines.append({
            "test_id": test_id,
            "body": action.get("body", ""),
            "cta": action.get("cta", "yes_stop"),
            "send_as": action.get("send_as", "vera"),
            "suppression_key": action.get("suppression_key", ""),
            "rationale": action.get("rationale", "")
        })
        print(f"Generated {test_id}")

    # Write to submission.jsonl
    with open("submission.jsonl", "w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            
    print("Done! submission.jsonl has been successfully updated.")

if __name__ == "__main__":
    main()
