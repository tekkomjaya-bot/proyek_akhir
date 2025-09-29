# Gunakan base image TensorFlow Serving
FROM tensorflow/serving:latest

# Buat direktori kerja
WORKDIR /app

# Copy model hasil training TFX
# Ganti '1758901692' dengan path versi model Anda jika berbeda
COPY ./outputs/serving_model/1758901692 /models/diabetes-model/1

# Copy file konfigurasi untuk monitoring
COPY ./config/prometheus.config /app/config/prometheus.config

# Copy skrip entrypoint
COPY ./scripts/tf_serving_entrypoint.sh /usr/bin/tf_serving_entrypoint.sh
RUN chmod +x /usr/bin/tf_serving_entrypoint.sh

# Set environment variables
ENV MODEL_NAME=diabetes-model
ENV MODEL_BASE_PATH=/models
# Path ke file konfigurasi monitoring di dalam container
ENV MONITORING_CONFIG_FILE=/app/config/prometheus.config

# Expose port untuk gRPC dan REST API (hanya untuk dokumentasi, Railway akan handle port)
# Port 8500 untuk gRPC, ${PORT} untuk REST API
EXPOSE 8500

# Jalankan server
ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]