FROM ghcr.io/lllyasviel/fooocus:edge

WORKDIR /content
USER root

COPY --chown=user:user . /content/app
RUN mv /content/app/models /content/app/models.org

COPY entrypoint_api.sh /content/
RUN chmod +x /content/entrypoint_api.sh && chown user:user /content/entrypoint_api.sh

USER user

CMD ["sh", "-c", "/content/entrypoint_api.sh ${CMDARGS}"]
