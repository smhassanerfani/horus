#! /usr/bin/bash

read -p "Image Name: " var
raspistill -o /home/pi/Downloads/cron_job_images/$var.jpg --width 1920 --height 1440
