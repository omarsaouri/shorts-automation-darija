#!/bin/bash
# Unloads and removes the three launchd jobs installed by install.sh.
set -euo pipefail

PLIST_DIR="$HOME/Library/LaunchAgents"

shopt -s nullglob
for plist in "$PLIST_DIR"/com.3larassi.*.plist; do
    launchctl unload "$plist" 2>/dev/null || true
    rm "$plist"
    echo "removed $(basename "$plist")"
done

echo "Scheduler uninstalled."
