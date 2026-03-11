#!/bin/bash
# Open Mind server kick-off
source ~/claude/env.sh

pkill -f om-server.py 2>/dev/null; sleep 1

nohup /home/jfischer/miniconda3/bin/python3 -u /home/jfischer/claude/_open-mind/om-server.py \
  >> /home/jfischer/claude/_open-mind/openmind.log 2>&1 &
disown $!
echo "om-server launched (PID $!)"

sleep 20
curl -s http://localhost:8250/api/stats | python3 -m json.tool | head
