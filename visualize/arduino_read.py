import serial

arduino = serial.Serial(port='/dev/tty.usbserial-1420', baudrate=115200, timeout=.1)
while True:
    data = arduino.readline() 
    print(data)