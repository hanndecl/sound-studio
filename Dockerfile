# syntax=docker/dockerfile:1

# resemble-enhance targets Python 3.10 and pins torch 2.1.x -> keep this base.
FROM python:3.10-slim-bookworm

# "cpu"  : small CPU-only torch wheels (default)
# "cuda" : default torch wheels (bundle CUDA 12.1 libs); run with --gpus all
ARG TORCH_VARIANT=cpu
# Download model weights at build time so containers start offline and fast
ARG PREFETCH_MODELS=1

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 build-essential git \
    && rm -rf /var/lib/apt/lists/*

# torch first, pinned to what resemble-enhance expects, so pip doesn't pull a second copy
RUN if [ "$TORCH_VARIANT" = "cpu" ]; then \
        pip install torch==2.1.1+cpu torchaudio==2.1.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip install torch==2.1.1 torchaudio==2.1.1; \
    fi

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Warm the model caches by running both engines once on a 2 s tone.
# (Resemble Enhance stores its weights inside site-packages, so this is the
#  only way they survive running the container as a non-root user.)
RUN if [ "$PREFETCH_MODELS" = "1" ]; then \
        mkdir -p /tmp/warm/in /tmp/warm/out_r /tmp/warm/out_d && \
        ffmpeg -loglevel error -f lavfi -i "sine=frequency=440:duration=2" -ar 44100 -ac 1 /tmp/warm/in/t.wav && \
        resemble_enhance /tmp/warm/in /tmp/warm/out_r --device cpu && \
        deepFilter /tmp/warm/in/t.wav -o /tmp/warm/out_d && \
        rm -rf /tmp/warm; \
    fi

# Script last: editing it doesn't invalidate the heavy layers above
COPY studio_dialogue.py /app/studio_dialogue.py
RUN chmod -R a+rX /app

# Mount your videos here; outputs are written next to the input file
WORKDIR /data
ENTRYPOINT ["python", "/app/studio_dialogue.py"]
CMD ["--help"]