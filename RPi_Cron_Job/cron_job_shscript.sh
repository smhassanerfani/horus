#! /usr/bin/bash

DATEDIR=$(date +"%Y-%m-%d");

if [ ! -d "/home/pi/Downloads/$DATEDIR" ]; then
	mkdir /home/pi/Downloads/$DATEDIR
fi

DATENOW=$(date +"%Y-%m-%d-%H%M");
FILE=/home/pi/Downloads/$DATEDIR/$DATENOW.jpg;


if [ ! -f "$FILE" ]; then
	python3 /home/pi/rf24libs/RF24/examples_linux/aava_trigger.py >> /home/pi/Downloads/$DATEDIR.txt;
	raspistill -n -o $FILE --width 1920 --height 1440;
fi
