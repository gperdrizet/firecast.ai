# Use slim buster python image
FROM python:3.7-slim-buster

WORKDIR /usr/lib/firecast.ai

# Install needed packages
COPY . ./
RUN ./install_packages.sh

# Run startup script
CMD ./start_up.sh