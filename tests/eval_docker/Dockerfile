# LSS eval image — for running benchmarks under constrained compute.
#
# Build:  docker build -t lss-eval -f tests/eval_docker/Dockerfile .
# Run:    docker run --rm --cpus=1 --memory=1g -e LSS_PROVIDER=local lss-eval python -m lss_cli eval
#
# The --cpus and --memory flags enforce hard cgroup limits, simulating
# cheap VPSes and throttled containers. Quality should be identical
# across profiles; only latency changes.

FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libc6-dev zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

COPY . /lss
WORKDIR /lss

RUN pip install --no-cache-dir '.[local]' ranx
