# Twilio SMS
export TWILIO_ACCOUNT_SID=REDACTED
export TWILIO_AUTH_TOKEN=REDACTED
export TWILIO_PHONE_NUMBER=REDACTED

export OPENAI_API_KEY=REDACTED

pkill -f "cloudflared tunnel run openmind" 2>/dev/null
pkill -f om-server.py 2>/dev/null
sleep 2
~/miniconda3/bin/python3 -u ~/openmind/om-server.py > openmind.log 2>&1 &
sleep 15
~/cloudflared tunnel run openmind > tunnel-om.log 2>&1 &
sleep 5
curl -s http://localhost:8250/api/stats | python3 -m json.tool | head
