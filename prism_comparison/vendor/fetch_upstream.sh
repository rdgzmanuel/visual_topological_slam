#!/bin/bash
# Fetch the PRISM-TopoMap sources at the pinned commit and apply the
# device-autodetect patch. The upstream repository carries no license, so its
# code is not redistributed here; this script reproduces the exact vendored
# tree used in the paper's experiments.
set -e
cd "$(dirname "$0")"
COMMIT=$(cut -d' ' -f1 UPSTREAM_COMMIT.txt)
if [ -d prism-topomap ]; then
    echo "vendor/prism-topomap already exists; delete it first to re-fetch."
    exit 1
fi
git clone https://github.com/kirillmouraviev/prism-topomap prism-topomap
cd prism-topomap
git checkout "$COMMIT"
rm -rf .git img
patch -p1 < ../device_autodetect.patch
cd ..
echo "Done: vendor/prism-topomap at $COMMIT with device patch applied."
