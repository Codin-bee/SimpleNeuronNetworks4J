package com.codingbee.snn4j.exceptions;

public class MethodCallingException extends Exception{
    public MethodCallingException(String message) {
        super("Method was passed wrong arguments" + message);
    }
}
