FROM tensorflow/tensorflow
COPY requirements.txt /root
RUN apt-get update
RUN apt-get install -y xvfb ffmpeg python3-opengl
RUN pip install -r /root/requirements.txt
RUN mkdir /.config /.local /.cache
RUN chmod uo+rw /.config /.local /.cache
RUN mkdir /workspace
RUN chmod uo+rw /workspace
CMD "/bin/bash"
