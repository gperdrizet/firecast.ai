# Use slim buster python image
FROM python:3.7-slim-buster

WORKDIR /usr/lib/firecast.ai

# Install needed packages
COPY . ./
RUN ./install_packages.sh

# Set timezone
ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Run startup script
CMD ./start_up.sh