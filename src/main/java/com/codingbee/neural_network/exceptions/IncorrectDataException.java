package com.codingbee.neural_network.exceptions;

public class IncorrectDataException extends Exception{
    public IncorrectDataException(String message) {
        super("Incorrect data inserted into:" + message);
    }
}
