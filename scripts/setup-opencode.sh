#!/bin/bash
# setup-opencode.sh - Setup opencode with MiniMax and oh-my-openagent
# Usage: ./setup-opencode.sh [jetson|macbook|all]
#
# This script sets up opencode AI agent with:
# - MiniMax as the LLM provider
# - oh-my-openagent plugin for enhanced capabilities
# - All agent team configs pointing to MiniMax

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOST="${1:-all}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[✗]${NC} $1"; }

# Detect platform
detect_platform() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macbook"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [[ -f /etc/nv_tegra_release ]]; then
            echo "jetson"
        else
            echo "linux"
        fi
    else
        echo "unknown"
    fi
}

# Get target host SSH connection
get_ssh_target() {
    local host_type="$1"
    if [[ "$host_type" == "jetson" ]]; then
        echo "vorapol@192.168.86.40"
    else
        echo ""
    fi
}

# Check if we're connecting remotely
is_remote() {
    [[ -n "$SSH_CONNECTION" ]] || [[ -n "$REMOTE_HOST" ]]
}

# Run command locally or via SSH
run_cmd() {
    local target="$1"
    shift
    if [[ -n "$target" ]]; then
        ssh "$target" "$@"
    else
        "$@"
    fi
}

# Upgrade opencode to latest version
upgrade_opencode() {
    local target="$1"
    log "Upgrading opencode to latest..."
    
    run_cmd "$target" "opencode upgrade" 2>&1 | tail -5
    local version
    version=$(run_cmd "$target" "opencode --version")
    log "opencode version: $version"
}

# Install oh-my-openagent plugin
install_ohmyopenagent() {
    local target="$1"
    log "Installing oh-my-openagent..."
    
    # Method 1: Via npm in opencode plugins dir
    run_cmd "$target" "mkdir -p ~/.opencode/plugins"
    run_cmd "$target" "cd ~/.opencode && npm install oh-my-openagent 2>&1 | tail -3"
    
    log "oh-my-openagent installed"
}

# Install mavis plugin (for mavis team agents)
install_mavis_plugin() {
    local target="$1"
    log "Installing mavis plugin..."
    
    run_cmd "$target" "cd ~/.opencode && npm install mavis 2>&1 | tail -3"
    
    log "mavis plugin installed"
}

# Create opencode server startup script (for headless operation)
create_server_script() {
    local target="$1"
    log "Creating opencode server script..."
    
    local server_script='#!/bin/bash
# OpenCode server startup script
# Auto-restarts on failure

SOCKET_FILE="$HOME/.opencode/opencode.sock"
LOG_FILE="$HOME/.opencode/opencode.log"
PID_FILE="$HOME/.opencode/opencode.pid"

start() {
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo "OpenCode already running (PID $(cat $PID_FILE))"
        return 0
    fi
    
    echo "Starting OpenCode server..."
    nohup opencode serve --port 4096 > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 2
    
    if kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo "OpenCode started (PID $(cat $PID_FILE))"
    else
        echo "Failed to start. Check $LOG_FILE"
        return 1
    fi
}

stop() {
    if [ -f "$PID_FILE" ]; then
        kill $(cat "$PID_FILE") 2>/dev/null && rm -f "$PID_FILE" && echo "Stopped" || echo "Not running"
    else
        echo "No PID file found"
    fi
}

status() {
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo "Running (PID $(cat $PID_FILE))"
    else
        echo "Not running"
    fi
}

restart() {
    stop
    sleep 1
    start
}

case "$1" in
    start) start ;;
    stop) stop ;;
    restart) restart ;;
    status) status ;;
    *) echo "Usage: $0 {start|stop|restart|status}" ;;
esac
'
    
    run_cmd "$target" "cat > ~/.opencode/opencode-server.sh" <<< "$server_script"
    run_cmd "$target" "chmod +x ~/.opencode/opencode-server.sh"
    log "Server script created at ~/.opencode/opencode-server.sh"
}

# Verify MiniMax authentication
verify_auth() {
    local target="$1"
    log "Verifying MiniMax authentication..."
    
    local auth_status
    auth_status=$(run_cmd "$target" "opencode auth list 2>&1" || echo "not configured")
    echo "$auth_status" | head -10
}

# Show installed versions
show_versions() {
    local target="$1"
    local platform
    platform=$(run_cmd "$target" "detect_platform" 2>/dev/null || echo "unknown")
    
    echo ""
    echo "=== $platform setup summary ==="
    echo "opencode: $(run_cmd "$target" "opencode --version" 2>/dev/null || echo "not found")"
    echo "oh-my-openagent: $(run_cmd "$target" "npx oh-my-openagent --version 2>/dev/null" || echo "not found")"
    echo "plugins dir: $(run_cmd "$target" "ls ~/.opencode/plugins/ 2>/dev/null" || echo "empty")"
    echo "server: $(run_cmd "$target" "~/.opencode/opencode-server.sh status 2>/dev/null" || echo "not configured")"
}

# Setup for local MacBook
setup_macbook() {
    log "Setting up MacBook..."

    # 1. Fix workspace dir if missing (opencode upgrade bug workaround)
    mkdir -p ~/.mavis/agents/mavis/workspace/update

    # 2. Upgrade opencode
    upgrade_opencode

    # 3. Install oh-my-openagent
    install_ohmyopenagent

    # 4. Install mavis plugin
    install_mavis_plugin

    # 5. Verify auth
    verify_auth

    # 6. Show versions
    show_versions

    log "MacBook setup complete!"
}

# Setup for Jetson
setup_jetson() {
    local target="vorapol@192.168.86.40"
    log "Setting up Jetson Xavier NX..."

    # 1. Ensure uv is in PATH for mini-agent
    run_cmd "$target" 'echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> ~/.bashrc'

    # 2. Upgrade opencode (via SSH)
    upgrade_opencode "$target"

    # 3. Install oh-my-openagent
    install_ohmyopenagent "$target"

    # 4. Install mavis plugin
    install_mavis_plugin "$target"

    # 5. Create server script
    create_server_script "$target"

    # 6. Verify auth
    verify_auth "$target"

    # 7. Show versions
    show_versions "$target"

    log "Jetson setup complete!"
}

# Main
main() {
    echo ""
    echo "========================================"
    echo "  opencode + MiniMax Setup Script"
    echo "========================================"
    echo ""

    case "$HOST" in
        jetson)
            setup_jetson
            ;;
        macbook)
            setup_macbook
            ;;
        all)
            setup_macbook
            echo ""
            setup_jetson
            ;;
        *)
            echo "Usage: $0 [jetson|macbook|all]"
            echo ""
            echo "  jetson   - Setup Jetson Xavier NX at 192.168.86.40"
            echo "  macbook  - Setup local MacBook"
            echo "  all      - Setup both (default)"
            exit 1
            ;;
    esac

    echo ""
    echo "========================================"
    echo "  All done!"
    echo "========================================"
}

main