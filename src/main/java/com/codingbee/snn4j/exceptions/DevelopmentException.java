package com.codingbee.snn4j.exceptions;

@SuppressWarnings("unused")
public class DevelopmentException extends UnsupportedOperationException {
    public DevelopmentException(String message) {
        super("The code has reached Exception that it should not be able to, this is due to inner bug," +
                " or because a feature has not been implemented properly yet: " + message);
    }
}