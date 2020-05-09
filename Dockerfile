FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

COPY ./requirements.txt ./

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY ./resnet50-19c8e357.pth ./root/.cache/torch/checkpoints/resnet50-19c8e357.pth
COPY ./app /app