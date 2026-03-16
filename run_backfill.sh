#!/bin/bash
# Requires OPENAI_API_KEY to be set in environment or loaded from env.sh
if [ -z "$OPENAI_API_KEY" ]; then
  source "$(dirname "$0")/../env.sh" 2>/dev/null || { echo 'OPENAI_API_KEY not set'; exit 1; }
fi
exec /home/jfischer/miniconda3/bin/python3 -u /home/jfischer/claude/_open-mind/backfill_enrichment.py
