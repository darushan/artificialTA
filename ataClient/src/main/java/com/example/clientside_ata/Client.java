package com.example.clientside_ata;

import javafx.scene.layout.VBox;

import java.io.*;
import java.net.Socket;

public class Client {
    private Socket socket;
    private BufferedReader bufferedReader;

    private BufferedWriter bufferedWriter;

    public Client(Socket socket)
    {
        try {
            this.socket = socket;
            this.bufferedReader = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            this.bufferedWriter = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream()));
        }
        catch (IOException e)
        {
            System.out.println("error printing the client");
            e.printStackTrace();
            closeEverything(socket,bufferedReader,bufferedWriter);
        }
    }

    public void sendMessageToServer(String messageToServer)
    {
        try {
            bufferedWriter.write(messageToServer);
            bufferedWriter.newLine();
            bufferedWriter.flush();
        }
        catch(IOException e){
            e.printStackTrace();
            System.out.println("Error Sending Message to the client");
            closeEverything(socket, bufferedReader, bufferedWriter);
        }
    }
    public void recieveMessageFromServer(VBox vBox)
    {
        new Thread(new Runnable() {
            @Override
            public void run() {
                while(socket.isConnected()){
                    try{
                        String messageFromClient = bufferedReader.readLine();

                        if(messageFromClient.equals("Thank you for using aTA") ){
                            closeEverything(socket,bufferedReader,bufferedWriter);
                            System.exit(0);
                        }


                        ClientController.addLabel(messageFromClient, vBox);
                    }
                    catch (IOException e){
                        e.printStackTrace();
                        System.out.println("Error receiving message from client");
                        closeEverything(socket, bufferedReader, bufferedWriter);
                        break;
                    }
                }
            }
        }).start();
    }

    public void closeEverything(Socket socket, BufferedReader bufferedReader, BufferedWriter bufferedWriter){
        try{
            if (bufferedReader != null){
                bufferedReader.close();
            }

            if (bufferedWriter != null){
                bufferedWriter.close();
            }

            if (socket != null){
                socket.close();
            }
        }
        catch (IOException e){
            e.printStackTrace();
        }
    }
}