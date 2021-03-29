FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime
WORKDIR /workspace/
RUN apt update && apt install build-essential -y
COPY requirements.txt /workspace/requirements.txt
COPY start.sh /workspace/start.sh
RUN pip install -r requirements.txt
RUN jt -t monokai -T -N
EXPOSE 4016
