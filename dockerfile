FROM python:3.11-bullseye
RUN apt-get update && apt-get install -y --no-install-recommends build-essential patchelf && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install "nuitka>=2.4" zstandard "numpy==1.26.4" "openai>=1.30.0" "pypdf2>=3.0.0"

CMD python -m nuitka \
  --onefile \
  --standalone \
  --enable-plugin=numpy \
  --follow-imports \
  --output-filename=kb \
  kb_fusion.py
