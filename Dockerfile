FROM tensorflow/tensorflow
COPY requirements.txt /root
RUN apt-get update
RUN apt-get install -y xvfb ffmpeg python3-opengl
RUN pip install -r /root/requirements.txt
CMD "/bin/bash"
