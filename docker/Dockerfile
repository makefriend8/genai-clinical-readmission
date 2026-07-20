


ARG CUDA_IMAGE=12.1.1-runtime-ubuntu22.04
FROM nvidia/cuda:${CUDA_IMAGE}


ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1


RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.9 python3.9-dev python3-pip python3-venv \
        build-essential git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.9 /usr/bin/python \
    && ln -sf /usr/bin/python3.9 /usr/bin/python3

WORKDIR /app


ARG UID=10001
RUN useradd --create-home --shell /bin/bash --uid "${UID}" appuser


COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install torch --index-url https://download.pytorch.org/whl/cu121 \
    && python -m pip install -r requirements.txt

# Copy the source code into the container. Raw/generated clinical data
# (cohort_final_v2.csv, mimic-iv-bhc.csv, data/processed, etc.) is excluded
# via .dockerignore and mounted in at runtime instead - see compose.yaml.
COPY . .

RUN mkdir -p data/processed experiments/output \
    && chown -R appuser:appuser /app

# Switch to the non-privileged user to run the application.
USER appuser

# This project is a sequence of scripts, not a web server, so there is
# nothing to EXPOSE. The default command runs the full pipeline in order:
# merge raw datasets -> preprocess/split -> baseline model -> LongT5 training.
# Override to run a single step instead, e.g.:
#   docker compose run --rm app python src/preprocess.py
#   docker compose run --rm app python scripts/evaluate.py
CMD ["bash", "-c", "python src/merge_datasets.py && python src/preprocess.py && python src/models/baseline_model.py && python scripts/train_longt5.py"]
