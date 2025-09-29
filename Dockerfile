FROM tensorflow/serving:latest

# Copy model hasil training TFX
COPY ./outputs/serving_model/1758901692 /models/diabetes-model/1
COPY ./config /model_config

# Set environment variables
ENV MODEL_NAME=diabetes-model
ENV MODEL_BASE_PATH=/models
ENV MONITORING_CONFIG="/model_config/prometheus.config"
# HAPUS BARIS INI: ENV PORT=8080
# Railway akan menyediakan variabel $PORT secara otomatis

# Entrypoint untuk TensorFlow Serving
# Skrip ini sudah benar karena menggunakan ${PORT} yang akan diisi oleh Railway
RUN echo '#!/bin/bash \n\n\
env \n\
tensorflow_model_server \
--port=8500 \
--rest_api_port=${PORT} \
--rest_api_host=0.0.0.0 \
--model_name=${MODEL_NAME} \
--model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
--monitoring_config_file=${MONITORING_CONFIG} \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh

ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]