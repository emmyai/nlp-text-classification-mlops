import json
import os

def load_parameters(config_path="config/parameters.json"):
    with open(config_path, "r") as f:
        return json.load(f)
