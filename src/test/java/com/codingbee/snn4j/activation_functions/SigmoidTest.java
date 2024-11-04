package com.codingbee.snn4j.activation_functions;

import org.junit.Test;
import org.junit.jupiter.api.Assertions;

public class SigmoidTest {
    @Test
    public void activateTest1(){
        double expected = 0.5;
        Sigmoid sigmoid = new Sigmoid();
        double actual = sigmoid.activate(0);
        Assertions.assertEquals(expected, actual);
    }
    @Test
    public void activateTest2(){
        double expected = 0.7310585786300049;
        Sigmoid sigmoid = new Sigmoid();
        double actual = sigmoid.activate(1);
        Assertions.assertEquals(expected, actual);
    }
    @Test
    public void activateTest3(){
        double expected = 0.2689414213699951;
        Sigmoid sigmoid = new Sigmoid();
        double actual = sigmoid.activate(-1);
        Assertions.assertEquals(expected, actual);
    }
}
