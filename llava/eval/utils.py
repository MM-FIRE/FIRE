import re
from datetime import datetime
import json
import os

def save(model_path, conv_mode, results, prefix=""):
    job_id = re.findall(r'\d+', model_path)[-1]
    output_path = f"data/eval/run_{datetime.now().strftime('%Y%m%d')}"
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, f"{prefix}_{conv_mode}_{job_id}.json")
    with open(output_path, "w") as f:
        json.dump(results, f)
        