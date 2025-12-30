# Building kb-fusion for Linux AMD64

This guide explains how to build kb-fusion for linux/amd64 platform using Docker.

## Quick Start

### Option 1: Build Wheel Only (Fast, ~30 seconds)

```bash
docker buildx build --platform linux/amd64 -f Dockerfile.simple -t kb-fusion-wheel .
docker run --platform linux/amd64 --rm -v "$PWD/dist-linux":/app kb-fusion-wheel
```

This creates a Python wheel file (`kb_fusion-0.1.1-py3-none-any.whl`) that can be installed with pip on any Linux AMD64 system.

### Option 2: Build Everything (Wheel + Standalone Binary, ~5-10 minutes)

```bash
./build-linux.sh
```

Or manually:

```bash
docker buildx build --platform linux/amd64 -t kb-fusion-build .
mkdir -p dist-linux
docker run --platform linux/amd64 --rm -v "$PWD/dist-linux":/app kb-fusion-build
```

This creates:
- `kb_fusion-0.1.1-py3-none-any.whl` - Python wheel package
- `kb-linux-amd64` - Standalone executable (Nuitka-compiled)

## Output

Build artifacts will be saved to `dist-linux/` directory.

### Wheel File

The wheel can be installed on any Linux AMD64 system with Python 3.9+:

```bash
pip install kb_fusion-0.1.1-py3-none-any.whl
```

### Standalone Binary

The Nuitka-compiled binary can be run directly without Python installed:

```bash
chmod +x kb-linux-amd64
./kb-linux-amd64 search /path/to/file "query"
```

## Requirements

- Docker Desktop or Docker with buildx support
- macOS (M1/M2/Intel) or Linux host
- ~2GB free disk space for build process

## Build Process

### What the Dockerfile Does

1. **Base Image**: Uses `python:3.11-slim` for minimal size
2. **System Dependencies**: Installs gcc, g++, patchelf for Nuitka compilation
3. **UV Installation**: Uses `uv` for fast dependency management
4. **Wheel Build**: Creates distributable Python package
5. **Nuitka Build**: Compiles standalone executable with all dependencies embedded

### Build Time

- Wheel only: ~30 seconds
- Full build (wheel + Nuitka): ~5-10 minutes (first time, cached afterwards)

## Troubleshooting

### Build Fails

1. Clean Docker cache:
   ```bash
   docker buildx prune -a
   ```

2. Rebuild from scratch:
   ```bash
   docker buildx build --no-cache --platform linux/amd64 -t kb-fusion-build .
   ```

### Permission Errors

The output files may be owned by root. Fix with:

```bash
sudo chown -R $USER:$USER dist-linux/
```

## Cross-Platform Notes

This builds for **linux/amd64** specifically. The resulting binaries will run on:
- x86_64 Linux systems (Ubuntu, Debian, RHEL, etc.)
- WSL2 on Windows
- Docker containers based on linux/amd64

They will NOT run on:
- ARM64 Linux (Raspberry Pi, AWS Graviton, etc.) - use `--platform linux/arm64`
- Windows native - requires separate Windows build
- macOS - requires separate macOS build



