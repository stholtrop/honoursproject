FROM tensorflow/tensorflow
COPY requirements.txt /root
RUN pip install -r /root/requirements.txt
CMD "/bin/bash"
