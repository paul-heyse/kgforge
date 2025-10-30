# syntax=docker/dockerfile:1

FROM python:3.13-slim AS builder

ENV UV_BIN=/opt/uv/bin
ENV PATH=${UV_BIN}:$PATH

RUN apt-get update \
    && apt-get install --no-install-recommends -y curl build-essential git \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p ${UV_BIN} \
    && curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --install-dir ${UV_BIN}

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY tools ./tools
RUN uv build

FROM python:3.13-slim AS runtime

ENV CATALOG_ROOT=/srv/catalog
WORKDIR /srv/app

COPY --from=builder /app/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/kgfoundry-*.whl \
    && rm -rf /tmp/kgfoundry-*.whl

COPY catalog_artifacts/ /srv/catalog/

ENTRYPOINT ["catalogctl"]
CMD ["--help"]
