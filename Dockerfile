FROM nvcr.io/nvidia/pytorch:22.11-py3

WORKDIR /app

COPY . /app/

ENV DEBIAN_FRONTEND noninteractive

RUN apt update
RUN apt install -y libsndfile1-dev ffmpeg
RUN pip install -e .

CMD ["python", "app.py"]
