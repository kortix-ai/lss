FROM python:3.11-bullseye

WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y \
    binutils \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY *.py ./
COPY README.md ./

# Install dependencies
RUN uv sync --frozen

# Install PyInstaller (MUCH faster than Nuitka - 2min vs 50min!)
RUN uv pip install pyinstaller

# Create output directory
RUN mkdir -p /output

# Build wheel package
RUN uv build --wheel --out-dir /output

# Build standalone executable with PyInstaller
RUN uv run pyinstaller \
    --onefile \
    --name lss-linux-amd64 \
    --distpath /output \
    --workpath /tmp/pyinstaller \
    --specpath /tmp \
    --clean \
    --noconfirm \
    lss_cli.py

# Default command: copy output to mounted volume
CMD ["sh", "-c", "cp -r /output/* /app/ && echo 'Build artifacts:' && ls -lh /app/"]
