import socket
import time

def check_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

print(f"Checking port 8000...")
for i in range(5):
    res = check_port(8000)
    print(f"Attempt {i+1}: {'OPEN' if res else 'CLOSED'}")
    time.sleep(1)
