#!/bin/bash
# ═══════════════════════════════════════════
# Open Mind — Setup & Launch Script
# ═══════════════════════════════════════════
# Run on c-jfischer3:
#   chmod +x setup.sh && ./setup.sh
# ═══════════════════════════════════════════

set -e

INSTALL_DIR="$HOME/openmind"
PORT=8250

echo "╔═══════════════════════════════════════╗"
echo "║         Open Mind — Setup             ║"
echo "╚═══════════════════════════════════════╝"

# Create directory structure
mkdir -p "$INSTALL_DIR/uploads"
echo "✓ Directory: $INSTALL_DIR"

# Copy files if running from a different directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$SCRIPT_DIR" != "$INSTALL_DIR" ]; then
    cp "$SCRIPT_DIR/om-server.py" "$INSTALL_DIR/"
    cp "$SCRIPT_DIR/om-viz.html" "$INSTALL_DIR/"
    [ -f "$SCRIPT_DIR/CLAUDE.md" ] && cp "$SCRIPT_DIR/CLAUDE.md" "$INSTALL_DIR/"
    [ -f "$SCRIPT_DIR/om-kickoff.sh" ] && cp "$SCRIPT_DIR/om-kickoff.sh" "$INSTALL_DIR/"
    echo "✓ Files copied to $INSTALL_DIR"
fi

# Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install --break-system-packages -q \
    fastapi uvicorn[standard] sentence-transformers \
    numpy httpx python-multipart 2>/dev/null || \
pip install -q \
    fastapi uvicorn[standard] sentence-transformers \
    numpy httpx python-multipart

echo "✓ Python packages installed"

# Check for tesseract (optional, for OCR)
if command -v tesseract &> /dev/null; then
    echo "✓ Tesseract OCR available"
else
    echo "⚠ Tesseract not found. Install with: sudo apt install tesseract-ocr"
    echo "  (OCR will be skipped for images without it)"
fi

# Check for OpenAI key
if [ -z "$OPENAI_API_KEY" ]; then
    echo ""
    echo "⚠ OPENAI_API_KEY not set. LLM features will be disabled."
    echo "  Set it with: export OPENAI_API_KEY=$OPENAI_API_KEY
    echo "  Or add to ~/.bashrc"
fi

# Create systemd service (optional)
read -p "Create systemd service for auto-start? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    SERVICE_FILE="$HOME/.config/systemd/user/openmind.service"
    mkdir -p "$(dirname "$SERVICE_FILE")"
    cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Open Mind Knowledge Graph
After=network.target

[Service]
Type=simple
WorkingDirectory=$INSTALL_DIR
ExecStart=$(which python3) om-server.py
Restart=always
RestartSec=5
Environment=OPENAI_API_KEY=${OPENAI_API_KEY:-}
Environment=OPENMIND_PORT=$PORT

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload
    systemctl --user enable openmind
    systemctl --user start openmind
    echo "✓ Systemd service created and started"
    echo "  Manage: systemctl --user {start|stop|restart|status} openmind"
    echo "  Logs:   journalctl --user -u openmind -f"
else
    echo ""
    echo "To start manually:"
    echo "  cd $INSTALL_DIR && python3 om-server.py"
fi

echo ""
echo "═══════════════════════════════════════"
echo " Open Mind running on port $PORT"
echo " Local:  http://localhost:$PORT"
echo " Tunnel: Set up cloudflared for remote access"
echo "═══════════════════════════════════════"
echo ""
echo "Cloudflared tunnel (if needed):"
echo "  cloudflared tunnel --url http://localhost:$PORT"
echo ""
echo "Twilio SMS webhook URL:"
echo "  https://your-tunnel-domain/api/sms"
echo ""
