# FROM ubuntu:latest
FROM python:3.8.1
COPY . /app
RUN ls -la /
WORKDIR /app
RUN pip3 install --no-cache-dir -r /app/requirements.txt
EXPOSE 5000
CMD ["python3", "/app/api/app.py"]