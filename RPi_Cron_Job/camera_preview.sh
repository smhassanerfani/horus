#! /usr/bin/bash

if [ ! -d "/home/pi/Downloads/cc" ]; then
	mkdir /home/pi/Downloads/cc
fi

read -p "Image Name: " var
raspistill -o /home/pi/Downloads/cc/$var.jpg --width 1920 --height 1440
