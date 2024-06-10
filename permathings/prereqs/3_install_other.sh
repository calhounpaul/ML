#!/bin/bash

sudo apt update
sudo apt install -y python3-pip python3-venv python3-dev jq

pip3 install --upgrade pip

pip3 install --upgrade docker

sudo sed -i 's/<policy domain="coder" rights="none" pattern="PDF" \/>//g' /etc/ImageMagick-6/policy.xml
