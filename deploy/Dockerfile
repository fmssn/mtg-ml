# syntax=docker/dockerfile:1
# Hosted play server image (docs/hosting.md). Build from the repository root:
#   docker build -f deploy/Dockerfile -t mtg-play:$(git rev-parse --short HEAD) \
#     --build-arg GIT_REV=$(git rev-parse HEAD) .
#
# Layout: the package is installed editable from /app so `mtg_ml/replay.py`
# finds its static files exactly as in a checkout (ROOT = /app: /app/viz,
# /app/apps/play). Weights are never part of the image; mount them at /models.

ARG PYTHON_IMAGE=python:3.11-slim-bookworm

########################################################################
# Builder: Rust toolchain + maturin build the native engine wheel; the
# runtime venv gets CPU-only torch, numpy and the `hosted` extra.
########################################################################
FROM ${PYTHON_IMAGE} AS builder

ARG RUST_TOOLCHAIN=stable
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    PATH=/opt/cargo/bin:$PATH

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential ca-certificates curl \
 && rm -rf /var/lib/apt/lists/* \
 && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --profile minimal --default-toolchain "${RUST_TOOLCHAIN}" \
 && rustc --version

# Build tool lives outside the runtime venv so it is not shipped.
RUN python -m pip install "maturin[patchelf]>=1.5,<2" \
 && python -m venv /opt/venv

# Third-party dependencies first (cached unless pyproject.toml changes).
# The `hosted` extra is read from pyproject.toml so it has one source of truth.
WORKDIR /app
COPY pyproject.toml ./
RUN /opt/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['optional-dependencies']['hosted']))" > /tmp/hosted-requirements.txt \
 && /opt/venv/bin/pip install numpy -r /tmp/hosted-requirements.txt

# Native engine. The crate embeds mtg_ml/engine/cards.toml at compile time.
COPY native/ /build/native/
COPY mtg_ml/engine/cards.toml /build/mtg_ml/engine/cards.toml
RUN cd /build/native \
 && maturin build --release --locked --interpreter /opt/venv/bin/python --out /build/wheels \
 && /opt/venv/bin/pip install /build/wheels/*.whl

# Application source, installed editable (no dependency resolution: done above).
COPY mtg_ml/ /app/mtg_ml/
COPY apps/ /app/apps/
COPY viz/ /app/viz/
COPY data/ /app/data/
RUN /opt/venv/bin/pip install --no-deps -e /app \
 && /opt/venv/bin/python -m compileall -q /app/mtg_ml \
 && /opt/venv/bin/python -c "import mtg_ml, mtg_ml_native; assert mtg_ml.__file__.startswith('/app/')"

########################################################################
# Runtime
########################################################################
FROM ${PYTHON_IMAGE} AS runtime

ARG GIT_REV=unknown
LABEL org.opencontainers.image.title="mtg-play" \
      org.opencontainers.image.description="mtg-ml hosted play server" \
      org.opencontainers.image.revision="${GIT_REV}"

RUN groupadd --system --gid 10001 mtg \
 && useradd --system --uid 10001 --gid 10001 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin mtg \
 && mkdir -p /models /data/state /data/replays \
 && chown 10001:10001 /data/state /data/replays

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

# Read-only root filesystem in production: nothing writes outside /tmp and /data.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    MTG_GIT_REV=${GIT_REV} \
    MTG_ENGINE=native \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MTG_HOSTED_HOST=0.0.0.0 \
    MTG_HOSTED_PORT=8080 \
    MTG_HOSTED_MODELS=/models \
    MTG_HOSTED_STATE=/data/state \
    MTG_HOSTED_REPLAYS=/data/replays

VOLUME ["/data/state", "/data/replays"]
WORKDIR /app
USER 10001:10001
EXPOSE 8080

# No curl in the image: probe /healthz with urllib (HTTP 503 raises, exit 1).
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('MTG_HOSTED_PORT', '8080'), timeout=4)"]

CMD ["python", "-m", "mtg_ml.hosted", "serve"]
