#!/bin/bash
# Load key from env.sh (never commit secrets)
source ~/claude/env.sh
exec ~/miniconda3/bin/python3 -u ~/claude/_open-mind/reenrich_diary.py
