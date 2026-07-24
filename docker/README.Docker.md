### What this runs

This is a clinical data research pipeline (data merge -> preprocess -> baseline model ->
LongT5 training), 

### Prerequisites

- Docker with the Compose plugin.
- An NVIDIA GPU with drivers + the [NVIDIA Container
  Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  installed (required by the `deploy.resources.reservations.devices` block in
  `compose.yaml`). Without a GPU, remove that block and expect LongT5
  training to be very slow.
- `cohort_final_v2.csv` and `mimic-iv-bhc.csv` placed in the repo root before
  running (see `../data/README.md` for how to obtain them).

### GPU passthrough

Before running the pipeline, confirm the host itself can reach the GPU from
a container:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi
```

If that fails, it's an NVIDIA Container Toolkit / driver problem on the
host, not something in this repo's Docker files - reinstall/reconfigure the
toolkit per NVIDIA's guide linked above before troubleshooting further.

The actual passthrough is the `deploy.resources.reservations.devices` block
in `compose.yaml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

- `capabilities` is the only required field - Compose refuses to start the
  service without it.
- `count: all` reserves every GPU on the host. If you have multiple GPUs and
  want to keep LongT5 off one that's in use elsewhere, use `device_ids:
  ["0"]` instead (find IDs via `nvidia-smi`) - `count` and `device_ids` are
  mutually exclusive.
- `NVIDIA_VISIBLE_DEVICES` and `NVIDIA_DRIVER_CAPABILITIES=compute,utility`
  are set explicitly in the `environment:` block. They're already the
  default in the `nvidia/cuda` base image, but are pinned here so
  passthrough doesn't silently break if the base image ever changes.

### VRAM / memory notes for LongT5 training

`scripts/train_longt5.py` runs `google/long-t5-tglobal-base` at a 4096-token
input length, which is heavy on VRAM even with `per_device_train_batch_size=1`
and `fp16` (both already set in that script). A few things this Docker setup
does to help:

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set in the
  container's environment. This reduces allocator fragmentation, which is
  what usually causes `CUDA out of memory` errors mid-run even when
  `nvidia-smi` shows free VRAM.
- `shm_size: "2gb"` gives the container's `/dev/shm` more room than Docker's
  64MB default. This doesn't matter today since the training script uses
  the default of 0 DataLoader workers, but it's cheap insurance if
  `dataloader_num_workers` is ever increased for throughput - PyTorch passes
  batches between worker processes through shared memory, and the default
  size causes crashes (not slowdowns) when it's too small.
- Watch VRAM live from the host while training with `watch -n1 nvidia-smi`,
  or from inside the container with `docker compose exec app nvidia-smi`.
- If you hit OOM anyway, the standard levers (in `scripts/train_longt5.py`)
  are: shrink `max_input_length`/`max_target_length` in
  `ClinicalNotesDataset`, or add `gradient_accumulation_steps` to
  `Seq2SeqTrainingArguments` rather than raising batch size.

### Building and running

From the `Docker/` directory:

```bash
docker compose up --build
```

By default this runs the full pipeline in order: `merge_datasets.py` ->
`preprocess.py` -> `baseline_model.py` -> `train_longt5.py`. The repo root is
bind-mounted into the container, so raw data you drop in and files the
pipeline generates (`merged_cohort_text_data.csv`, `data/processed/`, model
checkpoints) are visible on your host, and code edits don't require a
rebuild.

To run a single step instead of the whole pipeline:

```bash
docker compose run --rm app python src/preprocess.py
docker compose run --rm app python scripts/evaluate.py
```

### Deploying your image elsewhere

Build it directly, e.g.: `docker build -f Dockerfile -t myapp ..` (run from
this directory; the build context is the repo root). If your target
architecture differs from your dev machine, add `--platform=linux/amd64` (or
`linux/arm64`) as needed. Then push it to your registry, e.g.
`docker push myregistry.com/myapp`.

### References
* [Docker's Python guide](https://docs.docker.com/language/python/)
* [Docker Compose GPU support](https://docs.docker.com/compose/gpu-support/)
