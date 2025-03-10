package com.codingbee.snn4j.interface_implementations.activation_functions;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class LeakyReLUTest {
    LeakyReLU leakyReLU = new LeakyReLU();

    @Test
    public void aboveZeroTest1(){
        float result = leakyReLU.activate(0.2f);
        float expected = 0.2f;
        Assertions.assertEquals(expected, result);
    }

    @Test
    public void aboveZeroTest2(){
        float result = leakyReLU.activate(452.43f);
        float expected = 452.43f;
        Assertions.assertEquals(expected, result);
    }

    @Test
    public void aboveZeroTest3(){
        float result = leakyReLU.activate(75.32f);
        float expected = 75.32f;
        Assertions.assertEquals(expected, result);
    }

    @Test
    public void zeroTest(){
        float result = leakyReLU.activate(0);
        float expected = 0;
        Assertions.assertEquals(expected, result);
    }

    @Test
    public void underZeroTest1(){
        leakyReLU.setAlpha(0.1f);
        float result = leakyReLU.activate(-12.33f);
        float expected = -1.233f;
        Assertions.assertEquals(expected, result);
    }

    @Test
    public void underZeroTest2(){
        leakyReLU.setAlpha(0.01f);
        float result = leakyReLU.activate(-582.7f);
        float expected = -5.827f;
        Assertions.assertEquals(expected, result);
    }

    @Test
    public void underZeroTest3(){
        leakyReLU.setAlpha(0.5f);
        float result = leakyReLU.activate(-0.563f);
        float expected = -0.2815f;
        Assertions.assertEquals(expected, result);
    }
}
