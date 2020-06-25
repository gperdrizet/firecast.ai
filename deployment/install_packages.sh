#!/bin/sh
export DEBIAN_FRONTEND=noninteractive
export APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=True

# Update the package listing, so we know what package exist:
apt-get update

# Install security updates:
apt-get -y upgrade
apt-get install -y  curl gnupg2

# Install tf serving without AVX
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add - 
apt-get update 
apt-get install -y tensorflow-model-server-universal

# Delete cached files we don't need anymore:
apt-get clean
rm -rf /var/lib/apt/lists/*

pip install luigi
pip install Flask
