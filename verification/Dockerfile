FROM debian:bookworm-slim@sha256:7b140f374b289a7c2befc338f42ebe6441b7ea838a042bbd5acbfca6ec875818

ARG ELAN_VERSION=4.2.3
ARG LEAN_VERSION=4.32.0

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

ENV ELAN_HOME=/opt/elan
ENV PATH=/opt/elan/bin:${PATH}

RUN curl --proto '=https' --tlsv1.2 --fail --silent --show-error \
        https://raw.githubusercontent.com/leanprover/elan/v${ELAN_VERSION}/elan-init.sh \
        -o /tmp/elan-init.sh \
    && sh /tmp/elan-init.sh -y --no-modify-path \
        --default-toolchain leanprover/lean4:v${LEAN_VERSION} \
    && /opt/elan/bin/elan toolchain install leanprover/lean4:v${LEAN_VERSION} \
    && rm /tmp/elan-init.sh \
    && chmod -R a+rX /opt/elan

WORKDIR /workspace
