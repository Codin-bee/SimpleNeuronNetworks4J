package com.codingbee.snn4j.exceptions;

@SuppressWarnings("unused")
public class MethodCallingException extends IllegalAccessException {
    public MethodCallingException(String message) {
        super("Method called when it should not be, or with incorrect arguments: " + message);
    }
}