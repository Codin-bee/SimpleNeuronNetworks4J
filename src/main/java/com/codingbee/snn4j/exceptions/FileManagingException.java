package com.codingbee.snn4j.exceptions;

import java.io.IOException;

public class FileManagingException extends IOException {
    public FileManagingException(String message) {
        super("Exception occurred, working with file system: " + message);
    }
}