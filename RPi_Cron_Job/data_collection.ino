/*
 * This code is to be uploaded to the base package or solar package. The EPM package uses EPM_data_collection.ino. 
 * This is the sketch that will run during deployment and take stage, temperature, pressure, humidity, and power
 * measurements. The data is saved on an SD card and collection occurs every five minutes as dictated by RTC alarms.
 * 
 * To change the data collection cycle interval: edit the const int time_interval
 * 
 * Written by Corinne Smith 
 * Date: Nov 2022
*/

#include <SD.h>                         // for SD card module
#include <SPI.h>                        // for SPI
#include <Wire.h>                       // for I2C
#include <HCSR04.h>                     // for the USS
#include <DS3232RTC.h>                  // for the RTC https://github.com/JChristensen/DS3232RTC
#include <avr/sleep.h>                  // for sleep mode
#include <Adafruit_INA219.h>            // for voltage monitor

const int RTCinterrupt = 2;             // RTC interrupt from sleep mode on digital pin 2


// HC-SR04 ----------------------------------------------------------------------------------------------
const int trigPin = 9;     
const int echoPin = 8;      
UltraSonicDistanceSensor distanceSensor(trigPin, echoPin);

// SD ---------------------------------------------------------------------------------------------------
const int pinCS = 10;

// INA219 -----------------------------------------------------------------------------------------------
Adafruit_INA219 ina219;

// controls ---------------------------------------------------------------------------------------------
const int LED = A3;
const int time_interval = 1;          // interval in which the minutes will be delayed. Default is five minutes
unsigned long prevTimeElapsed = 0;

void setup() {
  Serial.begin(9600);
  pinMode(pinCS, OUTPUT);
  pinMode(LED, OUTPUT);
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  pinMode(RTCinterrupt, INPUT_PULLUP);            // configure the interrupt pin using the built-in pullup resistor
  Serial.println("Executing data_collection.ino");

  // SD card initialization --------------------------------------------------------------------------------------------------------------------------
  if (!SD.begin())                               
  {
    digitalWrite(LED, HIGH);                      // LED remains on if SD card does not work
    Serial.println("No SD card found!");
    while (1);                                    
  }
  else
  {
    Serial.println("SD card found!");
  }

  // RTC initializaiton ------------------------------------------------------------------------------------------------------------------------------
  RTC.begin();
  
  // initialize the alarms, clear the alarm flags, clear the alarm interrupts
  RTC.setAlarm(ALM1_MATCH_DATE, 0, 0, 0, 1);
  RTC.setAlarm(ALM2_MATCH_DATE, 0, 0, 0, 1);
  RTC.alarm(ALARM_1);
  RTC.alarm(ALARM_2);
  RTC.alarmInterrupt(ALARM_1, false);
  RTC.alarmInterrupt(ALARM_2, false);
  RTC.squareWave(SQWAVE_NONE);
  
  // get the current time from the RTC and set an alarm according to the time interval
  time_t t;                               
  t = RTC.get();                            
  
  // uncomment to set the time interval units to seconds
  // if(second(t) < 60 - time_interval){
  //   RTC.setAlarm(ALM1_MATCH_SECONDS , second(t) + time_interval, 0, 0, 0);
  // }
  // else {
  //   RTC.setAlarm(ALM1_MATCH_SECONDS , second(t) - 60 + time_interval, 0, 0, 0);
  // }
  
  // uncomment to set the time interval units to minutes
  if (minute(t) < 60 - time_interval) {
    RTC.setAlarm(ALM1_MATCH_MINUTES , 0, minute(t) + time_interval, 0, 0);
  }
  else {
    RTC.setAlarm(ALM1_MATCH_MINUTES , 0, minute(t) - 60 + time_interval, 0, 0);
  }

  // clear the alarm flag and configure the interrupt operation
  RTC.alarm(ALARM_1);
  RTC.squareWave(SQWAVE_NONE);
  RTC.alarmInterrupt(ALARM_1, true);

  // Voltage regulator initialization -----------------------------------------------------------------------------------------------------------------
  uint32_t currentFrequency;
  ina219.begin();
}

void loop() {
  digitalWrite(LED, LOW);     // turn off LED before sleeping
  delay(10);
  goSleep();
}

void goSleep() {
  Serial.println("Going to sleep...");
  delay(100);

  // activate sleep mode, attach interrupt and assign a waking function to run
  sleep_enable();                               
  attachInterrupt(digitalPinToInterrupt(RTCinterrupt), RTCtrigger, LOW);
  set_sleep_mode(SLEEP_MODE_PWR_DOWN);              // set to full sleep mode   
  sleep_cpu();                                  

  // run the data collection function after the waking function
  logData();                                   
  
  // set the next alarm
  time_t t;
  t = RTC.get();
  
  // uncomment to set the time interval units to seconds
  // if(second(t) < 60 - time_interval){
  //   RTC.setAlarm(ALM1_MATCH_SECONDS , second(t) + time_interval, 0, 0, 0);
  // }
  // else {
  //   RTC.setAlarm(ALM1_MATCH_SECONDS , second(t) - 60 + time_interval, 0, 0, 0);
  // }
  
  // uncomment to set the time interval units to minutes
  if (minute(t) < 60 - time_interval) {
    RTC.setAlarm(ALM1_MATCH_MINUTES , 0, minute(t) + time_interval, 0, 0);
  }
  else {
    RTC.setAlarm(ALM1_MATCH_MINUTES , 0, minute(t) - 60 + time_interval, 0, 0);
  }

  // clear the alarm flag
  RTC.alarm(ALARM_1);
}

void RTCtrigger() {
  // this is the wake up function to run once the RTC interrupt is fired
  Serial.println("RTC interrupt fired");
  delay(100);
  sleep_disable();                        // disable sleep mode
  detachInterrupt(digitalPinToInterrupt(RTCinterrupt));          // clear the interrupt flag
}

void logData() {
  // this is the data collection function
  unsigned long timeElapsed = millis();
  Serial.println("Recording data...");
  digitalWrite(LED, HIGH);
  delay(10);

  // take five distance readings and take the average
  int water_level[5];
  float total_water_level = 0;
  float counter = 0;
  for (int i = 0; i < 5; i++) 
  {
      water_level[i] = distanceSensor.measureDistanceCm();
      delay(100);
      if (water_level[i] > 0)
      {
          total_water_level += water_level[i];
          counter++;
      }
  }

  float avgDist = (float)total_water_level / counter;
  
  // record the voltage, current, and power draw from the LiPo
  float busvoltage = ina219.getBusVoltage_V();
  float current_mA = ina219.getCurrent_mA();
  float power_mW = ina219.getPower_mW();

  // print the data to the file that will be saved on the SD card
  File myFile = SD.open("001.csv", FILE_WRITE);    // change to the file name you want to store the data

  if (myFile)                             // tests if the file has opened
  {
    // write the RTC data
    time_t t = RTC.get();
    myFile.print(String(year(t)));
    myFile.print('-');
    myFile.print(String(month(t)));
    myFile.print('-');
    myFile.print(String(day(t)));
    myFile.print(',');
    myFile.print(String(hour(t)));
    myFile.print(':');
    myFile.print(String(minute(t)));
    myFile.print(':');
    myFile.print(String(second(t)));
    myFile.print(',');

    // write the USS data
    for (int i = 0; i < 5; i++) {
      myFile.print(water_level[i]); 
      myFile.print(',');
    }

    myFile.print(avgDist);

    // write the power data
    myFile.print(busvoltage);
    myFile.print(',');
    myFile.print(current_mA);
    myFile.print(',');
    myFile.print(power_mW);
    myFile.print(',');

    myFile.println();
    myFile.close();           // closes and saves the file to the SD card
    
    Serial.print("Complete! Elapsed time: "); Serial.print(timeElapsed - prevTimeElapsed); Serial.println(" ms");
    prevTimeElapsed = timeElapsed;
  }
  else
  {
    Serial.println("Error in opening file.");
    digitalWrite(LED, HIGH);                // LED will stay on if the file is not opening properly. 
    while (1);
  }
  delay(100);
  digitalWrite(LED, LOW);
}
