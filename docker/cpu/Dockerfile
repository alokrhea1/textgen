# BUILDER
FROM ubuntu:22.04
WORKDIR /builder
ARG BUILD_EXTENSIONS="${BUILD_EXTENSIONS:-}"
ARG APP_UID="${APP_UID:-6972}"
ARG APP_GID="${APP_GID:-6972}"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,rw \
    apt update && \
    apt install --no-install-recommends -y git vim build-essential python3-dev pip bash curl && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /home/app/
COPY . /home/app/textgen
RUN mkdir -p /home/app/textgen/user_data && touch /home/app/textgen/user_data/CMD_FLAGS.txt
WORKDIR /home/app/textgen
RUN GPU_CHOICE=N LAUNCH_AFTER_INSTALL=FALSE INSTALL_EXTENSIONS=TRUE ./start_linux.sh --verbose
EXPOSE ${CONTAINER_PORT:-7860} ${CONTAINER_API_PORT:-5000}
# set umask to ensure group read / write at runtime
WORKDIR /home/app/textgen
CMD umask 0002 && export HOME=/home/app/textgen && ./start_linux.sh --listen
