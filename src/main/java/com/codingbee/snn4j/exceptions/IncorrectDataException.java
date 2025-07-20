package com.codingbee.snn4j.exceptions;

public class IncorrectDataException extends IllegalArgumentException{
    public IncorrectDataException(String message) {
        super("Incorrect data found or argument passed: " + message);
    }
}