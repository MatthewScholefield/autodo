#!/usr/bin/env bash

cd "$(dirname "$0")"
set -e

# === Setup Virtualenv ===
[[ -f .venv/ ]] || python3 -m venv .venv/
source .venv/bin/activate

# === Setup Mask R-CNN ===
[[ -d compile/Mask_RCNN ]] || git clone https://github.com/matterport/Mask_RCNN.git compile/Mask_RCNN

cd compile/Mask_RCNN
pip install -r requirements.txt
python setup.py install


