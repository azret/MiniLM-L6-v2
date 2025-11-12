import sys, os

import bert

def _load_model_by_name(name: str):
    assert name == "all-MiniLM-L12-v2", f"Model name mismatch: {name}"
    return bert._load_model_by_name(
        name,
        os.path.dirname(__file__)
    )
