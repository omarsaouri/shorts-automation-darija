#!/bin/bash
# Installs the three launchd jobs (ingest/publisher/reporter) into
# ~/Library/LaunchAgents and loads them. Not run automatically by anything —
# this is a deliberate, manual step (see docs/progress.md's scheduler section for
# why: starting this for real means unattended YouTube uploads on a cron).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLIST_DIR="$HOME/Library/LaunchAgents"
mkdir -p "$PLIST_DIR" "$REPO_DIR/logs"

for plist in "$REPO_DIR"/scheduler/com.3larassi.*.plist; do
    name="$(basename "$plist")"
    cp "$plist" "$PLIST_DIR/$name"
    launchctl unload "$PLIST_DIR/$name" 2>/dev/null || true
    launchctl load "$PLIST_DIR/$name"
    echo "loaded $name"
done

echo
echo "Scheduler installed. Jobs are armed but RunAtLoad is false, so nothing"
echo "fires until the next scheduled interval. Check status:"
echo "  launchctl list | grep 3larassi"
echo "Logs land in $REPO_DIR/logs/"
