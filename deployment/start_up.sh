#!/bin/sh

# nohup tensorflow_model_server \
#   --rest_api_port=8501 \
#   --model_name=parallel_LSTM \
#   --model_base_path=/usr/lib/firecast.ai/trained_models >server.log 2>&1

# %%bash --bg 
tensorflow_model_server \
  --rest_api_port=8501 \
  --model_name=parallel_LSTM \
  --model_base_path=/home/siderealyear/springboard/capstone/wildfire_production/deployment/trained_models/ >server.log 2>&1