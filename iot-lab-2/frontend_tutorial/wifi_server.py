import socket
import picar_4wd as fc
import time

HOST = "0.0.0.0" # IP address of your Raspberry PI
PORT = 65432          # Port to listen on (non-privileged ports are > 1023)
DATA = b""
finalData = b""
value = ""

fc.servo.set_angle(0)
time.sleep(1)

def startSocketConnection():
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()

        try:
            while 1:
                client, clientInfo = s.accept()
                print("server recv from: ", clientInfo)
                while 1:
                    DATA = client.recv(1024)      # receive 1024 Bytes of message in binary format
                    if DATA != b"":
                        value = DATA.decode().strip()+","+str(fc.utils.power_read())+","+str(fc.utils.cpu_temperature())
                        finalData = str.encode(value)
                        if DATA == b"87\r\n":
                            fc.forward(10) 
                        elif DATA == b"83\r\n":
                            fc.backward(10)
                        elif DATA == b"65\r\n":
                            fc.turn_left(10)
                        elif DATA == b"68\r\n":
                            fc.turn_right(10)
                        elif DATA == b"stop\r\n":
                            fc.stop()
                        client.sendall(finalData) # Echo back to client
        except: 
            print("Closing socket")
            client.close()
            s.close()
            
if __name__ == '__main__':
    startSocketConnection()
    
    
