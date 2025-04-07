FROM python:3.10

SHELL ["/bin/bash", "-c"]
RUN apt-get update -qq && apt-get upgrade -qq &&\
    apt-get install -qq man wget sudo vim tmux 

RUN yes | pip install --upgrade pip

COPY requirements.txt /home/
WORKDIR /home
RUN yes | pip install -r requirements.txt


COPY ".env" /home/
COPY src/config.yaml /home/
COPY src/data_random_sample.py /home/
COPY src/generate_opro_training_examples.py /home/
COPY src/load_hf_data.py /home/
COPY src/opro_scoring.py /home/
COPY src/opro.py /home/
COPY src/scoring_new_prompts.py /home/
COPY src/scoring.py /home/