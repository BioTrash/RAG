FROM archlinux:latest

RUN pacman -Syu --noconfirm ca-certificates bash procps && pacman -Scc --noconfirm

WORKDIR /

COPY server_XPU.sh /server_XPU.sh
RUN chmod +x /server_XPU.sh

ENTRYPOINT ["/server_XPU.sh"]