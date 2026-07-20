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
