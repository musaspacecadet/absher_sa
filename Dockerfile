# Pull base image.
FROM jlesage/baseimage-gui:alpine-3.19-v4

ENV PYTHONUNBUFFERED=1
# Set the name of the application.
RUN set-cont-env APP_NAME "chromium"

# Install Chrome dependencies and Chrome
RUN apk add --no-cache \
    chromium \
    python3 \
    py3-pip 

# Copy the start script.
COPY startapp.sh /startapp.sh
COPY requirements.txt /requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /requirements.txt --break-system-packages

COPY . /