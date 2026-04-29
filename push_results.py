"""
Push daily results to Hugging Face dataset.
"""

import json
import pandas as pd
from huggingface_hub import HfApi, HfFileSystem
import config

def push_daily_result(result_dict):
    """Append or replace JSON file in HF dataset."""
    filename = f"emd_hybrid_{config.TODAY}.json"
    # We'll upload as a new file each day (or overwrite if same date run again)
    api = HfApi()
    fs = HfFileSystem(token=config.HF_TOKEN)
    # Convert to JSON string
    json_str = json.dumps(result_dict, indent=2)
    # Upload
    with fs.open(f"datasets/{config.HF_OUTPUT_REPO}/{filename}", "w") as f:
        f.write(json_str)
    print(f"Results pushed to {config.HF_OUTPUT_REPO}/{filename}")
