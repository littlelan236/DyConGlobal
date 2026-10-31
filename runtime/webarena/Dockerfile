# Browser worker image is intentionally independent of the runner's Python 3.12.
# Build from the frozen ServiceNow/webarena-verified and web-arena-x/webarena
# source checkouts at the commits recorded in its publication manifest.
ARG PYTHON_IMAGE=python:3.10-slim-bookworm
FROM ${PYTHON_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /opt/webarena
ARG HTTP_PROXY
ARG HTTPS_PROXY
ENV HTTP_PROXY=${HTTP_PROXY} HTTPS_PROXY=${HTTPS_PROXY}
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium libasound2 libatk1.0-0 libatk-bridge2.0-0 libatspi2.0-0 libcairo2 \
    libcups2 libdbus-1-3 libdrm2 libgbm1 libglib2.0-0 libnspr4 libnss3 \
    libpango-1.0-0 libx11-6 libx11-xcb1 libxcb1 libxcomposite1 libxdamage1 \
    libxext6 libxfixes3 libxkbcommon0 libxrandr2 libxshmfence1 libxss1 \
    libxtst6 fonts-liberation ca-certificates && rm -rf /var/lib/apt/lists/*
COPY browser-requirements.txt /opt/webarena/browser-requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/webarena/browser-requirements.txt
COPY upstream/browser_env /opt/webarena/browser_env
COPY worker.py /opt/webarena/worker.py
COPY browser-selftest.py /opt/webarena/browser-selftest.py
ENV PYTHONPATH=/opt/webarena DYCON_BROWSER_OUTPUT=/output
# The rootless daemon maps this UID to the owning host user, allowing the
# attempt-owned output bind mount to be written without making it world-writable.
USER 0:0
EXPOSE 8765
ENTRYPOINT ["python", "-u", "/opt/webarena/worker.py"]
