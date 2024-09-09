#!/bin/bash

####################
# These prereq scripts are used to prep a new ubuntu 24.04 VM
####################

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

