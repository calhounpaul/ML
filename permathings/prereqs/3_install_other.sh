#!/bin/bash

sudo apt update
sudo apt install -y python3-pip python3-venv python3-dev jq imagemagick git-lfs

pip3 install --upgrade pip

pip3 install --upgrade docker selenium pillow hwid aes-cipher
sudo apt install -y calibre jq

sudo sed -i 's/<policy domain="coder" rights="none" pattern="PDF" \/>//g' /etc/ImageMagick-6/policy.xml
