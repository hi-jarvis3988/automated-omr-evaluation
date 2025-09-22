FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev
RUN pip install --no-cache-dir -r requirement.txt
EXPOSE 5000
CMD ["python", "app.py"]