FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Set the working directory
WORKDIR /workspace

ARG HTTP_PROXY
ARG http_proxy
ARG HTTPS_PROXY
ARG https_proxy
ARG USER_ID
ARG GROUP_ID

ENV HTTP_PROXY=${HTTP_PROXY}
ENV http_proxy=${http_proxy}
ENV HTTPS_PROXY=${HTTPS_Proxy}
ENV https_proxy=${https_proxy}


# Create a user with the provided user and group ID
RUN if [ -n "${USER_ID}" ] && [ -n "${GROUP_ID}" ]; then \
    groupadd -g ${GROUP_ID} myuser && \
    useradd -u ${USER_ID} -g ${GROUP_ID} -m myuser; \
    fi

RUN usermod -s /bin/bash myuser

# # Install sudo
RUN apt-get update && apt-get install -y sudo

# Add myuser to sudo group
RUN usermod -aG sudo myuser

# Set up passwordless sudo for myuser
RUN echo "myuser ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/myuser && chmod 0440 /etc/sudoers.d/myuser