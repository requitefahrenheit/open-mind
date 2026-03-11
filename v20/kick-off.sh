source ~/claude/env.sh
# Twilio SMS
export TWILIO_ACCOUNT_SID=$TWILIO_ACCOUNT_SID
export TWILIO_AUTH_TOKEN=$TWILIO_AUTH_TOKEN
export TWILIO_PHONE_NUMBER=$TWILIO_PHONE_NUMBER

export OPENAI_API_KEY=$OPENAI_API_KEY

pkill -f "cloudflared tunnel run openmind" 2>/dev/null
pkill -f om-server.py 2>/dev/null
sleep 2
~/miniconda3/bin/python3 -u ~/claude/open-mind/om-server.py > openmind.log 2>&1 &
sleep 15
~/cloudflared tunnel run openmind > tunnel-om.log 2>&1 &
sleep 5
curl -s http://localhost:8250/api/stats | python3 -m json.tool | head
