import sys
import socket
from aTA_Model import ATA_Model

HOST = '' # Server private ip
SOCKET_LIST = []
RECV_BUFFER = 4096
PORT = 4206

ACK_TEXT = 'text_received'

def chat_server():
    aTA = ATA_Model()
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    new_host = socket.gethostbyname(HOST)
    server_socket.bind((new_host, PORT))
    server_socket.listen(10)

    # add server socket object to the list of readable connections
    SOCKET_LIST.append(server_socket)

    print("Chat server started on port " + str(PORT))

    while True:

        # get the list sockets which are ready to be read through select
        # 4th arg, time_out  = 0 : poll and never block
        (clientConnected, clientAddress) = server_socket.accept()

        print(f"Accepted a connection request from {clientAddress[0]}:{clientAddress[1]}\n")

        while True:
            request = clientConnected.recv(1024)
            decoded_request = request.decode()

            print("Request:")
            print(decoded_request, end='')

            if decoded_request == "exit\r\n":
                clientConnected.send("Thank you for using aTA\n".encode())
                server_socket.close()
                exit()
            
            message = "Working on getting you an answer...\n"
            clientConnected.send(message.encode())

            answer = aTA.get_answer(query=decoded_request)
            answer += "\n" # Newline required.

            print("Response:")
            print(answer)

            clientConnected.send(answer.encode())

if __name__ == "__main__":
    sys.exit(chat_server())
