package com.codingbee.snn4j.neural_network.network;

import com.codingbee.snn4j.exceptions.IncorrectDataException;
import com.codingbee.snn4j.neural_network.MLP;
import org.junit.Test;
import org.junit.jupiter.api.Assertions;

public class MLPConstructorTest {
    @Test
    public void constructorTest1(){
        Exception actual = Assertions.assertThrows(IncorrectDataException.class, () -> new MLP(0, 1, null, "network"));
        IncorrectDataException expected = new IncorrectDataException("MLP constructor - input layer size must be higher than zero");
        Assertions.assertEquals(expected.getMessage(), actual.getMessage());
    }
    @Test
    public void constructorTest2(){
        Exception actual = Assertions.assertThrows(IncorrectDataException.class, () -> new MLP(1, 0, null, "network"));
        IncorrectDataException expected = new IncorrectDataException("MLP constructor - output layer size must be higher than zero");
        Assertions.assertEquals(expected.getMessage(), actual.getMessage());
    }
    @Test
    public void constructorTest3(){
        Exception actual = Assertions.assertThrows(IncorrectDataException.class, () -> new MLP(1, 1, new int[]{0}, "network"));
        IncorrectDataException expected = new IncorrectDataException("MLP constructor - all hidden layers sizes must be higher than zero");
        Assertions.assertEquals(expected.getMessage(), actual.getMessage());
    }
    @Test
    public void constructorTest4(){
        Exception actual = Assertions.assertThrows(IncorrectDataException.class, () -> new MLP(1, 1, null, ""));
        IncorrectDataException expected = new IncorrectDataException("MLP constructor - MLP name must not be an empty String");
        Assertions.assertEquals(expected.getMessage(), actual.getMessage());
    }
    @Test
    public void constructorTest5(){
        new MLP(1,1,new int[] {1}, "network") ;
    }
}
