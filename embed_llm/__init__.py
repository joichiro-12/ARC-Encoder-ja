import os

# Each path must end with "/". They can also be given by environment variables.

# Paths towards ARC-Encoders either yours or released ones
TMP_PATH = os.environ.get("ARC_TMP_PATH", "")

# Path towards base models and backbones
MODEL_PATH = os.environ.get("ARC_MODEL_PATH", "")

DATA_PATH = os.environ.get("ARC_DATA_PATH", "")
