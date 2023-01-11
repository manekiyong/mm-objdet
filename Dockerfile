FROM nvcr.io/nvidia/pytorch:20.12-py3

ENV DEBIAN_FRONTEND noninteractive
# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

ADD build /build
WORKDIR /build
RUN make

RUN apt-get install -y --no-install-recommends ffmpeg

ADD /src /src

WORKDIR /src

RUN cd /src/yolo/mish-cuda && python setup.py build install


# RUN python lumapi/initialize_db.py

# CMD ["uvicorn", "main:api", "--host", "0.0.0.0"]
