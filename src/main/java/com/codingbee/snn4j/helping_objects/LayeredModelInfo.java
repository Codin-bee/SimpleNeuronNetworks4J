package com.codingbee.snn4j.helping_objects;

import com.codingbee.snn4j.exceptions.FileManagingException;
import com.codingbee.snn4j.interface_implementations.layers.FullyConnectedLayer;
import com.codingbee.snn4j.interfaces.model.Layer;
import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class LayeredModelInfo {
    private String dir;
    private List<Layer> layers;

    public LayeredModelInfo(String path) throws FileManagingException {
        try {
            ObjectMapper mapper = new ObjectMapper();
            mapper.setVisibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY);
            mapper.readerForUpdating(this).readValue(new File(path), LayeredModelInfo.class);
        } catch (IOException e) {
            throw new FileManagingException("An Exception occurred while trying to init the " +
                    "LayeredModelInfo from file" + path + ": " + e.getLocalizedMessage());
        }
    }

    public void save(String path) throws FileManagingException {
        try {
            ObjectMapper mapper = new ObjectMapper();
            mapper.writeValue(new File(path), this);
        } catch (IOException e) {
            throw new FileManagingException("An Exception occurred trying to save the values of " +
                    "the FullConnectedLayer: " + path + ": " + e.getLocalizedMessage());
        }
    }


    public LayeredModelInfo(String dir, List<Layer> layers) {
        this.dir = dir;
        this.layers = layers;
    }

    public String getDir() {
        return dir;
    }

    public void setDir(String dir) {
        this.dir = dir;
    }

    public List<Layer> getLayers() {
        return layers;
    }

    public void setLayers(List<Layer> layers) {
        this.layers = layers;
    }
}
