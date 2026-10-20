# Backend image: API, run worker and watchlist scheduler (same code, different commands).
# Build context is the repo root (the evaluation runner reads briefs/).
FROM python:3.12-slim

# pango/harfbuzz: WeasyPrint PDF export. libgl/glib: Docling's image pipeline.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz0b libffi8 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1

WORKDIR /app/backend
# Dependencies first (cached unless the lock changes). Note: PyTorch (via Docling) makes this layer large.
COPY backend/pyproject.toml backend/uv.lock backend/.python-version ./
RUN uv sync --frozen --no-dev --no-install-project
COPY backend/ ./
COPY briefs/ /app/briefs/
RUN uv sync --frozen --no-dev

EXPOSE 8000
CMD ["uvicorn", "insightforge.main:app", "--host", "0.0.0.0", "--port", "8000"]
