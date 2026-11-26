# Sandbox for GPU tests of PRs from people who aren't maintainers (ADR-0010, docs/gpu-check.md).
#
# Built once by the maintainer, by hand, on the WSL distro's own Docker engine (never Docker
# Desktop):
#     docker --context default build -t brotkit-gpu-sandbox:1 tools/sandbox
# `tools/gpu_check.py --reviewed --sandbox docker <PR>` runs it twice per Python version:
#   1. `uv sync` with network, the PR's source mounted read-only, writing only to its venv
#      and a per-run uv cache;
#   2. `pytest -m gpu` with `--network none`, the GPU, the venv mounted read-only and one
#      writable output directory for the JUnit file.
# The image holds no credentials, and gpu_check never mounts HOME, ~/.config/gh, SSH keys,
# the Docker socket or /mnt/c into it. Containers run as the maintainer's uid with every
# capability dropped, no-new-privileges and a read-only root filesystem.
FROM ghcr.io/astral-sh/uv:0.13.9@sha256:d3b601f0c7b118bc1753febf7c41cc80981df678445e0254284eecf84db79a7d AS uv

FROM ubuntu:24.04@sha256:562111b88caa79c71391c0c4d3c2114a7d5c35915c20561fd893cb1e07929e6b

COPY --from=uv /uv /usr/local/bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Interpreters are baked in, so the runs never download one. The venvs gpu_check creates
# point at these paths, which is why phase 2 can use phase 1's venv.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN uv python install 3.14 3.15 && chmod -R a+rX /opt/uv-python
ENV UV_PYTHON_DOWNLOADS=never \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

USER 1000:1000
