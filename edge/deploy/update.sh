#!/usr/bin/env bash
# Run on the Pi to install the latest main:  /opt/e3/edge/deploy/update.sh
set -euo pipefail
cd /opt/e3
git fetch origin main
git reset --hard origin/main
cd edge
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -e ".[pi]"
# Install/authenticate GitHub CLI (gh) for the user running this script first.
# The downloader checks and installs YAMNet plus the classifier/model-settings pair.
# A failed download or validation exits here, before restarting the service.
.venv/bin/python deploy/download_models.py --sudo-install
sudo systemctl restart e3-edge
echo "Updated to $(git rev-parse --short HEAD)"
