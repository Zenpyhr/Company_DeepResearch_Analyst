FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY app /app/app
COPY agents /app/agents
COPY graph /app/graph
COPY pipelines /app/pipelines
COPY prompts /app/prompts
COPY schemas /app/schemas
COPY skills /app/skills
COPY storage /app/storage
COPY streamlit_app.py /app/streamlit_app.py

RUN pip install --upgrade pip && pip install -e .

EXPOSE 8080

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
