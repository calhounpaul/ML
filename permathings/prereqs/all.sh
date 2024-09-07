#!/bin/bash

if [ "$(lspci | grep -i nvidia)" ]; then
  echo "Nvidia GPU detected"
  bash ./0_*.sh
  bash ./1_*.sh
  bash ./2_*.sh
  bash ./3_*.sh
  sudo reboot
else
  echo "No Nvidia GPU detected"
fi

