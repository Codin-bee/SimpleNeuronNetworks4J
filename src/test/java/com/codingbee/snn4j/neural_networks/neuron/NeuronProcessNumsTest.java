package com.codingbee.snn4j.neural_networks.neuron;

import com.codingbee.snn4j.exceptions.IncorrectDataException;
import com.codingbee.snn4j.neural_networks.mlp.Neuron;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assertions;

public class NeuronProcessNumsTest {
    //Functionality tests
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
    //Edge case: too long array
    @Test
    public void processNumsTest4(){
        Neuron neuron = new Neuron(new double[]{2, 1, 4}, 3);
        IncorrectDataException actual = Assertions.assertThrows(IncorrectDataException.class, () -> neuron.processNums(new double[]{1, 5, 7, 3}));
        IncorrectDataException expected = new IncorrectDataException("Neuron - processing numbers - the array is longer than the amount of weights the network was initialized with");
        Assertions.assertEquals(expected.getLocalizedMessage(), actual.getLocalizedMessage());
    }
}
