package com.codingbee.snn4j.neural_network.neuron;

import com.codingbee.snn4j.neural_network.Neuron;
import org.junit.Test;
import org.junit.jupiter.api.Assertions;

public class NeuronProcessNumsTest {
    @Test
    public void processNumsTest1(){
        Neuron neuron = new Neuron(new double[]{1, 2, 3}, 4);
        neuron.processNums(new double[]{1, 2, 3});
        double actual = neuron.getFinalValue();
        double expected = 18;
        Assertions.assertEquals(expected, actual);
    }
    @Test
    public void processNumsTest2(){
        Neuron neuron = new Neuron(new double[]{3, 4}, 5);
        neuron.processNums(new double[]{1, 2});
        double actual = neuron.getFinalValue();
        double expected = 16;
        Assertions.assertEquals(expected, actual);
    }
    @Test
    public void processNumsTest3(){
        Neuron neuron = new Neuron(new double[]{6, 7}, 9);
        neuron.processNums(new double[]{5, 8});
        double actual = neuron.getFinalValue();
        double expected = 95;
        Assertions.assertEquals(expected, actual);
    }
}
