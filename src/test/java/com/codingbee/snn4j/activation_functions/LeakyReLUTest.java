package com.codingbee.snn4j.activation_functions;

import org.junit.Test;
import org.junit.jupiter.api.Assertions;

public class LeakyReLUTest {
    @Test
    public void activateTest1(){
        double expected = 1;
        LeakyReLU leakyReLU = new LeakyReLU();
        leakyReLU.setAlpha(0.01);
        double actual = leakyReLU.activate(1);
        Assertions.assertEquals(expected, actual);
    }
    @Test
    public void activateTest2(){
        double expected = 0;
        LeakyReLU leakyReLU = new LeakyReLU();
        leakyReLU.setAlpha(0.01);
        double actual = leakyReLU.activate(0);
        Assertions.assertEquals(expected, actual);
    }
    @Test
    public void activateTest3(){
        double expected = -0.01;
        LeakyReLU leakyReLU = new LeakyReLU();
        leakyReLU.setAlpha(0.01);
        double actual = leakyReLU.activate(-1);
        Assertions.assertEquals(expected, actual);
    }
    @Test
    public void activateTest4(){
        double expected = -0.1;
        LeakyReLU leakyReLU = new LeakyReLU();
        leakyReLU.setAlpha(0.1);
        double actual = leakyReLU.activate(-1);
        Assertions.assertEquals(expected, actual);
    }
}
