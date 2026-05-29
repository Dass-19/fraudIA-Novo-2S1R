FROM python:3.13.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1

WORKDIR /usr/src/app

RUN pip install --no-cache-dir uv

COPY requirements.txt ./

RUN uv pip install --system --no-cache -r requirements.txt

COPY . .

CMD ["uvicorn", "scripts.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
