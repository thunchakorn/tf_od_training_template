FROM tf_od

WORKDIR /home/tensorflow
COPY requirements.txt ./
RUN python -m pip install -r requirements.txt