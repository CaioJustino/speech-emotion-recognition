"""
    Module src: Speech Emotion Recognition (SER) Package.
    Modular MLOps architecture for SER using CNNs and CRNNs.
"""

__version__ = "1.0.0"

import os
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

_yaml_path = os.path.join(BASE_DIR, "config.yaml")
if os.path.exists(_yaml_path):
    with open(_yaml_path, "r") as _f:
        _cfg = yaml.safe_load(_f)

    for _k, _v in _cfg.get("paths", {}).items():
        globals()[_k.upper()] = os.path.join(BASE_DIR, _v) if _v else BASE_DIR
        
    for _section in ["audio_processing", "dataset", "training", "hardware"]:
        for _k, _v in _cfg.get(_section, {}).items():
            globals()[_k.upper()] = _v
