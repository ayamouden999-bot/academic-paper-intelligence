"""
view_logs.py — Print a summary of all pipeline runs.
Usage: python view_logs.py
"""

import os, json, glob
from config import LOGS_DIR

log_files = sorted(glob.glob(os.path.join(LOGS_DIR, "*.json")))

if not log_files:
    print("No logs found.")
else:
    print(f"\n{'='*60}")
    print(f"  PIPELINE RUN LOGS ({len(log_files)} entries)")
    print(f"{'='*60}\n")

    runs = {}
    for f in log_files:
        try:
            with open(f) as fp:
                entry = json.load(fp)
            ts   = entry.get("timestamp", "")[:19]
            agent = entry.get("agent", "unknown")
            action = entry.get("action", "unknown")

            # Group by date
            date = ts[:10]
            if date not in runs:
                runs[date] = []
            runs[date].append((ts, agent, action, entry))
        except:
            continue

    for date, entries in sorted(runs.items()):
        print(f"  📅 {date}")
        for ts, agent, action, entry in entries:
            time = ts[11:]
            print(f"     {time} | {agent:<25} | {action}")
            # Print key details for important actions
            if action == "classification_complete":
                print(f"              Field: {entry.get('field')} | Novelty: {entry.get('novelty')}")
            if action == "decision":
                approved = "✅ Approved" if entry.get("approved") else "❌ Rejected"
                print(f"              Human decision: {approved}")
            if action == "pipeline_complete":
                print(f"              Output: {entry.get('output', 'N/A')}")
        print()