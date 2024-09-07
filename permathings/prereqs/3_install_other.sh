#!/bin/bash

sudo apt update
sudo apt install -y python3-full python3-pip python3-venv jq imagemagick git-lfs tmux axel git calibre

pip3 install --upgrade pip

pip3 install --upgrade docker selenium pillow hwid aes-cipher

#sudo sed -i 's/<policy domain="coder" rights="none" pattern="PDF" \/>//g' /etc/ImageMagick-6/policy.xml
