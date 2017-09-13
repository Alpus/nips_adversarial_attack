FROM gcr.io/tensorflow/tensorflow:1.3.0-devel-gpu-py3
MAINTAINER Alexander Pushin <work@apushin.com>

COPY requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt
