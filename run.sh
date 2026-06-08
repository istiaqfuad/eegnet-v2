#!/bin/bash
set -e
cd "$(dirname "$0")"
MODEL="${1:-freqaware}"
python3 main.py --model "$MODEL"
