/*
* This code is to be uploaded to the Arduino Nano in the ultrasonic package.
* Ultrasonic distance readings are continuously transmitted using the nRF24L01 2.4 GHz wireless module every 
* 5 seconds to a listening raspberry pi
* 
* To operate with Serial communication: 
* 1. Plug in the Nano and open the Arduino IDE
* 2. Go to Tools > Port > and select the correct COM port
* 3. Open the Serial monitor by clocking on the magnifying glass button in the top right corner of the IDE
* 4. Change the baud rate in the drop down box in the bottom of the Serial monitor from 9600 baud to 115200 baud
* 5. You should then see Serial output in the monitor of the distance values being sent
* 
* 
* Adapted from Anuj Dev's blog: https://medium.com/@anujdev11/communication-between-arduino-and-raspberry-pi-using-nrf24l01-818687f7f363
* 
* Written by Corinne Smith May 9 2022
* Modified by Mohammad Erfani Sep 2022
*/

#include <SPI.h>                // serial peripheral interface library
#include <SD.h>                 // SD card library
#include "nRF24L01.h"           // nRF24L01 library
#include "RF24.h"               // nRF24L01 library https://github.com/nRF24/RF24
#include <NewPing.h>            // ultrasonic sensor library https://bitbucket.org/teckel12/arduino-new-ping/wiki/Home
#include <Wire.h>               // I2C library
#include "RTClib.h"             // real-time clock library https://github.com/adafruit/RTClib

// define HC-SR04 parameters
#define TRIGGER_PIN 9      
#define ECHO_PIN  8     
#define MAX_DIST 400

// construct ultrasonic sensor
NewPing sonar(TRIGGER_PIN, ECHO_PIN, MAX_DIST);

// construct nRF24L01
RF24 radio(7, 6);

// Radio pipe addresses for the 2 nodes to communicate.
const uint64_t pipes[2] = {0xF0F0F0F0E1LL, 0xF0F0F0F0D2LL};

// initialise the clock
RTC_DS3231 rtc;

// set transmission interval in ms
int send_interval = 5000;

void setup() {
	// begin serial communication
	Serial.begin(115200);
	Serial.println("Executing horus_transmitter.ino");

	// initialize SD card
	if (!SD.begin()) {
	Serial.println("SD card did not begin.");
	while (1);
	}

	// begin the clock
	rtc.begin();

	// uncomment to set the date and time of the RTC to when the sketch was compiled (only if the clock has never been initialized)
	//rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));

	// initialize radio
	radio.begin();

	// Set the trasmit power to highest available to increase to the logest range we can achieve
	radio.setPALevel(RF24_PA_MAX);

	// Set the speed of trasmission to the quickest/ longest available
	// RF24_250KBPS for 250kbs, RF24_1MBPS for 1Mbps, or RF24_2MBPS for 2Mbps
	radio.setDataRate(RF24_250KBPS);

	// Set RF communication channel (0-127).
	radio.setChannel(124);

	radio.setAutoAck(true);
	radio.enableDynamicPayloads();
	radio.setRetries(5, 15);

	radio.openWritingPipe(pipes[0]);
	radio.openReadingPipe(1, pipes[1]);

	// set module as a receiver (ensures nothing is sent during this time)
	radio.startListening();
}

void loop() {

  // This is what we receive from the other device (the transmitter)
  unsigned char data;

  // Is there any data for us to get?
  if (radio.available()) {

    // Go and read the data and put it into that variable
    while (radio.available()) {
      radio.read(&data, sizeof(char));
    }

    // get the current time
    DateTime now = rtc.now();

    // take five distance readings and take the average
    int water_level[5];
    float total_water_level = 0;
    float counter = 0;
    for (int i = 0; i < 5; i++) 
    {
		water_level[i] = sonar.ping_cm();
		delay(100);
		if (water_level[i] > 0)
		{
			total_water_level += water_level[i];
			counter++;
		}
    }
    
    float avgDist = (float)total_water_level / counter;
    
    Serial.print("Sending Data: ");
    for (int i = 0; i < 5; i++)
	{
		Serial.print(water_level[i]);
		Serial.print(", ");
	}
    Serial.print("---> Average: ");
    Serial.print(avgDist);
    Serial.println(" cm");
    delay(100);

    // convert the distance to a character array for sending
    String water_level_str = String(avgDist);
    static char send_payload[50];
    water_level_str.toCharArray(send_payload, 50);
    // Serial.println(send_payload);

    // set module as a transmitter
    radio.stopListening();

    // send the distance reading
    Serial.print("Now sending length: ");
    Serial.println(sizeof(send_payload));
    radio.write(send_payload, sizeof(send_payload));

    // Now, resume listening so we catch the next packets.
    radio.startListening();

    // save time stamp and distance reading to SD card
    saveData(now, water_level, avgDist);

    // delay loop for the transmission interval
    delay(send_interval);

    }
}

void saveData(DateTime now, int water_level[], float avgDist){
	
	// define and open file to save data
	File myFile = SD.open("001.csv", FILE_WRITE);
    // if the file opens, print the time stamp and distance reading
    if (myFile) {
		// write date and time
		myFile.print(now.year(), DEC);
		myFile.print('-');
		myFile.print(now.month(), DEC);
		myFile.print('-');
		myFile.print(now.day(), DEC);
		myFile.print(',');
		myFile.print(now.hour(), DEC);
		myFile.print(':');
		myFile.print(now.minute(), DEC);
		myFile.print(':');
		myFile.print(now.second(), DEC);
		myFile.print(',');

		// write the Ultra Sonic data
		for (int i = 0; i < 5; i++)
		{
			myFile.print(water_level[i]); 
			myFile.print(',');
		}
      
		myFile.print(avgDist); 

		myFile.println();
		myFile.close();
    }
    
	// if the file does not open, print a message to the Serial monitor
    else {
		Serial.println("File did not open!");
	}
}
