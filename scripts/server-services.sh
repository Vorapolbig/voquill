#!/usr/bin/env bash
# Manage Voquill home server services.
#
# Service configs live in scripts/services/ — edit them there, then run
# `install` to apply changes. This is the single source of truth.
#
# Usage: ./server-services.sh <command> [service]
#
# Commands:
#   install           Copy service files to systemd, enable and start all services
#   restart [svc]     Restart one service or all if none specified
#   status            Show status of all services
#   logs <svc>        Follow logs for a service (Ctrl-C to stop)
#
# Services: whisper-gpu, llama-server, diarize
#
# Notes:
#   - llama-server base unit is installed by apt/snap; only the override is managed here
#   - Secrets loaded by diarize service from ~/.whisper-secrets (HF_TOKEN)
#   - CF credentials for transcribe.sh client are in ~/.whisper.env

set -euo pipefail

SERVICES_DIR="$(cd "$(dirname "$0")/services" && pwd)"
SERVICES=(whisper-gpu llama-server diarize)

usage() {
  sed -n '/^# Usage:/,/^# Notes:/p' "$0" | sed 's/^# \?//'
  exit 1
}

require_sudo() {
  if [[ $EUID -ne 0 ]]; then
    echo "This command requires sudo. Re-running with sudo..." >&2
    exec sudo "$0" "$@"
  fi
}

cmd_install() {
  require_sudo install "$@"

  echo "==> Installing whisper-gpu.service"
  cp "$SERVICES_DIR/whisper-gpu.service" /etc/systemd/system/whisper-gpu.service

  echo "==> Installing llama-server override"
  mkdir -p /etc/systemd/system/llama-server.service.d
  cp "$SERVICES_DIR/llama-server-override.conf" /etc/systemd/system/llama-server.service.d/override.conf

  echo "==> Installing diarize.service"
  cp "$SERVICES_DIR/diarize.service" /etc/systemd/system/diarize.service

  echo "==> Reloading systemd"
  systemctl daemon-reload

  for svc in "${SERVICES[@]}"; do
    echo "==> Enabling and starting $svc"
    systemctl enable "$svc"
    systemctl restart "$svc"
  done

  echo ""
  cmd_status
}

cmd_restart() {
  local target="${1:-}"
  if [[ -n "$target" ]]; then
    echo "==> Restarting $target"
    sudo systemctl restart "$target"
  else
    for svc in "${SERVICES[@]}"; do
      echo "==> Restarting $svc"
      sudo systemctl restart "$svc"
    done
  fi
  echo ""
  cmd_status
}

cmd_status() {
  for svc in "${SERVICES[@]}"; do
    printf "%-20s" "$svc:"
    status=$(systemctl is-active "$svc" 2>/dev/null || true)
    case "$status" in
      active)   printf "\033[32m%s\033[0m" "$status" ;;
      failed)   printf "\033[31m%s\033[0m" "$status" ;;
      *)        printf "\033[33m%s\033[0m" "$status" ;;
    esac
    printf "\n"
  done
}

cmd_logs() {
  local target="${1:-}"
  [[ -z "$target" ]] && { echo "Usage: $0 logs <service>" >&2; exit 1; }
  journalctl -u "$target" -f
}

case "${1:-}" in
  install)  shift; cmd_install "$@" ;;
  restart)  shift; cmd_restart "${1:-}" ;;
  status)   cmd_status ;;
  logs)     shift; cmd_logs "${1:-}" ;;
  *)        usage ;;
esac
