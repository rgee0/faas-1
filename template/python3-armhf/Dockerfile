FROM armhf/python:3.6-alpine

RUN echo "http://dl-4.alpinelinux.org/alpine/edge/community" >> /etc/apk/repositories

RUN apk --no-cache add curl make automake gcc g++ subversion python3-dev musl-dev libffi-dev lapack-dev gfortran \
    && curl -sSL https://github.com/openfaas/faas/releases/download/0.8.2/fwatchdog-armhf > /usr/bin/fwatchdog \
    && chmod +x /usr/bin/fwatchdog
    && apk del curl --no-cache

# Add non root user
RUN addgroup -S app && adduser -S -g app app
RUN chown app /home/app

USER app
ENV PATH=$PATH:/home/app/.local/bin

WORKDIR /home/app/

COPY index.py           .
COPY requirements.txt   .
RUN pip install --user app  -r requirements.txt

RUN mkdir -p function
RUN touch ./function/__init__.py

WORKDIR /home/app/function/
COPY function/requirements.txt	.
RUN pip install --user app -r requirements.txt

WORKDIR /home/app/
COPY function           function

ENV fprocess="python3 index.py"

HEALTHCHECK --interval=1s CMD [ -e /tmp/.lock ] || exit 1

CMD ["fwatchdog"]
