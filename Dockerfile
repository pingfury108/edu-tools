FROM python:3.12

COPY ./ /app

WORKDIR /app
RUN pip install --no-cache-dir --upgrade -r /app/requirements.lock

CMD ["fastapi", "run", "src/edu_tools/api.py", "--port", "8000",  "--host", "0.0.0.0"]
