package com.codingbee.snn4j.activation_functions;

import org.junit.Test;
import org.junit.jupiter.api.Assertions;

public class ELUTest {
    @Test
    public void activateTest1(){
        double expected = -0.6321205588285577;
        ELU elu = new ELU();
        elu.setAlpha(1);
        double actual = elu.activate(-1);
        Assertions.assertEquals(expected, actual);
    }
    @Test
    public void activateTest2(){
        double expected = 0;
        ELU elu = new ELU();
        elu.setAlpha(1);
        double actual = elu.activate(0);
        Assertions.assertEquals(expected, actual);
    }
    @Test
    public void activateTest3(){
        double expected = 1;
        ELU elu = new ELU();
        elu.setAlpha(1);
        double actual = elu.activate(1);
        Assertions.assertEquals(expected, actual);
    }
    @Test
    public void activateTest4(){
        double expected = 1;
        ELU elu = new ELU();
        elu.setAlpha(2);
        double actual = elu.activate(1);
        Assertions.assertEquals(expected, actual);
    }
    @Test
    public void activateTest5(){
        double expected = 0;
        ELU elu = new ELU();
        elu.setAlpha(2);
        double actual = elu.activate(0);
        Assertions.assertEquals(expected, actual);
    }
    @Test
    public void activateTest6(){
        double expected = -1.2642411176571153;
        ELU elu = new ELU();
        elu.setAlpha(2);
        double actual = elu.activate(-1);
        Assertions.assertEquals(expected, actual);
    }
}
