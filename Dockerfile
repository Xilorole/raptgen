FROM python:3.7

WORKDIR /root

COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get install -y libcairo2
