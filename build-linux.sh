#!/bin/bash
set -e

echo "Building kb-fusion for linux/amd64..."

# Build the Docker image
docker buildx build --platform linux/amd64 -t kb-fusion-build .

# Create output directory
mkdir -p dist-linux

# Run the container and extract artifacts
docker run --platform linux/amd64 --rm -v "$PWD/dist-linux":/app kb-fusion-build

echo ""
echo "Build complete! Artifacts saved to dist-linux/"
ls -lh dist-linux/



