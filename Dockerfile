FROM nvidia/cuda:11.4.1-base-ubuntu20.04

RUN apt-get update && apt-get install -y python3 && apt-get install -y python3-pip