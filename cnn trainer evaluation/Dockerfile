FROM tensorflow/tensorflow:latest-gpu-py3

RUN mkdir /App
WORKDIR /App
ADD . .
RUN pip install keras matplotlib pandas
CMD ["python","k04_evaluate_epoch_with_log_dist.py"]
