FROM gemmafc-gfootball:latest

RUN python3 -m pip install --no-cache-dir \
    fastapi==0.115.6 \
    uvicorn==0.32.1 \
    cerebras-cloud-sdk \
    together

WORKDIR /gemmafc
COPY app ./app

EXPOSE 8000

CMD ["sh", "-c", "xvfb-run -a -s \"-screen 0 1280x720x24\" python3 -m uvicorn app.server:app --host 0.0.0.0 --port 8000"]
