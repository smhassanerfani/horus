import time
from RF24 import RF24, RF24_PA_MAX, RF24_1MBPS, RF24_250KBPS
import RPi.GPIO as GPIO
from datetime import datetime

def millis():
    return round(time.time() * 1000)

if __name__ == "__main__":
    radio = RF24(22, 0)
    pipes = [0xF0F0F0F0E1, 0xF0F0F0F0D2]

    radio.begin()

    radio.setPALevel(RF24_PA_MAX)
    radio.setDataRate(RF24_250KBPS)
    radio.setChannel(124)

    radio.setAutoAck(True)
    radio.enableDynamicPayloads()
    radio.setRetries(5, 15)

    radio.openWritingPipe(pipes[1])
    radio.openReadingPipe(1, pipes[0])

    buffer = bytearray(1)

    radio.stopListening()

    # Trigger Ultrasonic Sensor
    if not radio.write(buffer):
        # print("No acknowledgment of transmission!")
        pass
    
    radio.startListening()

    started_waiting_at = millis()
    
    while not radio.available():
        waiting_time = millis() - started_waiting_at
        if millis() - started_waiting_at > 1000:
            print(f"{datetime.now().strftime('%m/%d/%Y %H:%M')}, No response received, timeout!")
            exit()

    # print(waiting_time)
    
    buffer = radio.read(radio.getDynamicPayloadSize())
    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M')}, {str(buffer, 'utf-8')}")
