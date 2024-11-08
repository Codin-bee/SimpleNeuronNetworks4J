package com.codingbee.snn4j.activation_functions;

import com.codingbee.snn4j.interface_implementations.activation_functions.ReLU;
import org.junit.Test;
import org.junit.jupiter.api.Assertions;

public class ReLUTest {
    @Test
    public void activateTest1(){
        double expected = 1;
        ReLU reLU = new ReLU();
        double actual = reLU.activate(1);
        Assertions.assertEquals(expected, actual);
    }
    @Test
    public void activateTest2(){
        double expected = 0;
        ReLU reLU = new ReLU();
        double actual = reLU.activate(0);
        Assertions.assertEquals(expected, actual);
    }
    @Test
    public void activateTest3(){
        double expected = 0;
        ReLU reLU = new ReLU();
        double actual = reLU.activate(-1);
        Assertions.assertEquals(expected, actual);
    }
}
