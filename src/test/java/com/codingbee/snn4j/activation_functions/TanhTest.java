package com.codingbee.snn4j.activation_functions;

import com.codingbee.snn4j.interface_implementations.activation_functions.Tanh;
import org.junit.Test;
import org.junit.jupiter.api.Assertions;

public class TanhTest {
    @Test
    public void activateTest1(){
        double expected = -0.7615941559557649;
        Tanh tanh = new Tanh();
        double actual = tanh.activate(-1);
        Assertions.assertEquals(expected, actual);
    }
    @Test
    public void activateTest2(){
        double expected = 0;
        Tanh tanh = new Tanh();
        double actual = tanh.activate(0);
        Assertions.assertEquals(expected, actual);
    }
    @Test
    public void activateTest3(){
        double expected = 0.7615941559557649;
        Tanh tanh = new Tanh();
        double actual = tanh.activate(1);
        Assertions.assertEquals(expected, actual);
    }
}
