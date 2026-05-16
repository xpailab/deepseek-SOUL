FROM python:3.11-slim

LABEL org.opencontainers.image.title="DeepSoul Agent"
LABEL org.opencontainers.image.description="下一代 AI Agent 框架"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ripgrep git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
COPY soul/ ./soul/
COPY skills/ ./skills/
COPY web/ ./web/

RUN pip install --no-cache-dir -e ".[all]"

ENV SOUL_HOME=/root/.soul
RUN mkdir -p $SOUL_HOME/workspace $SOUL_HOME/skills $SOUL_HOME/logs

EXPOSE 18789 8080

ENTRYPOINT ["python", "-m", "soul.cli"]
CMD ["chat"]
