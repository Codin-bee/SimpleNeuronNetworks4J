package com.codingbee.snn4j.exceptions;

public class MethodCallingException extends IllegalArgumentException{
    public MethodCallingException(String message) {
        super("Method was passed wrong arguments" + message);
    }
}
