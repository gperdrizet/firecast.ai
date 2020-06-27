#!/bin/sh

nohup tensorflow_model_server \
  --rest_api_port=8501 \
  --model_name=parallel_LSTM \
  --model_base_path=/usr/lib/firecast.ai/trained_models