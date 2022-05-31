#! /usr/bin/bash

DATEDIR=$(date +"%Y-%m-%d");

if [ ! -d "/home/pi/Downloads/$DATEDIR" ]; then
	mkdir /home/pi/Downloads/$DATEDIR
fi

python3 /home/pi/rf24libs/RF24/examples_linux/aava_trigger.py >> /home/pi/Downloads/$DATEDIR.txt;

DATENOW=$(date +"%Y-%m-%d-%H%M");

raspistill -n -o /home/pi/Downloads/$DATEDIR/$DATENOW.jpg --width 1920 --height 1440
