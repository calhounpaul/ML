#!/bin/bash

sudo apt update
sudo apt install -y python3-full python3-pip python3-venv jq imagemagick git-lfs tmux axel git calibre

pip3 install --upgrade pip

pip3 install --upgrade pillow aes-cipher


pip3 install hwid --break-system-packages
pip3 install --upgrade huggingface_hub wget docker selenium --break-system-packages

#sudo sed -i 's/<policy domain="coder" rights="none" pattern="PDF" \/>//g' /etc/ImageMagick-6/policy.xml
