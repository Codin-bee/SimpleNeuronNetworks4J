package com.codingbee.snn4j.interface_implementations.activation_functions;


import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class ReLUTest {
    ReLU reLU = new ReLU();

    @Test
    public void aboveZeroTest(){
        float result = reLU.activate(0.2f);
        float expected = 0.2f;
        Assertions.assertEquals(expected, result);
    }

    @Test
    public void aboveZeroTest2(){
        float result = reLU.activate(452.43f);
        float expected = 452.43f;
        Assertions.assertEquals(expected, result);
    }

    @Test
    public void aboveZeroTest3(){
        float result = reLU.activate(75.32f);
        float expected = 75.32f;
        Assertions.assertEquals(expected, result);
    }

    @Test
    public void zeroTest(){
        float result = reLU.activate(0);
        float expected = 0;
        Assertions.assertEquals(expected, result);
    }

    @Test
    public void underZeroTest(){
        float result = reLU.activate(-12.33f);
        float expected = 0;
        Assertions.assertEquals(expected, result);
    }

    @Test
    public void underZeroTest2(){
        float result = reLU.activate(-56382.95f);
        float expected = 0;
        Assertions.assertEquals(expected, result);
    }

    @Test
    public void underZeroTest3(){
        float result = reLU.activate(-0.563f);
        float expected = 0;
        Assertions.assertEquals(expected, result);
    }
}
