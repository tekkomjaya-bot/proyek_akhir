FROM prom/prometheus:latest

# Ubah baris ini untuk menyertakan path dari root proyek
COPY prometheus/prometheus.yml /etc/prometheus/prometheus.yml

EXPOSE 9090

# Perintah untuk menjalankan Prometheus dengan file konfigurasi Anda
CMD ["--config.file=/etc/prometheus/prometheus.yml"]