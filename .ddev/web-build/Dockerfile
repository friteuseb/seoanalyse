ARG BASE_IMAGE
FROM $BASE_IMAGE

RUN apt-get update && \
    apt-get install -y php8.2-redis --no-install-recommends && \
    phpenmod redis && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


