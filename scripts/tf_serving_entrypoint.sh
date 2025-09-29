#!/bin/bash

# Menjalankan TensorFlow Model Server
# --port=8500 untuk gRPC
# --rest_api_port=${PORT} untuk REST API, ${PORT} diisi otomatis oleh Railway
# --monitoring_config_file menunjuk ke file konfigurasi Prometheus

tensorflow_model_server \
  --port=8500 \
  --rest_api_port=${PORT} \
  --rest_api_host=0.0.0.0 \
  --model_name=${MODEL_NAME} \
  --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
  --monitoring_config_file=${MONITORING_CONFIG_FILE} \
  "$@"