package com.codingbee.neural_network;

import com.codingbee.exceptions.IncorrectDataException;
import org.junit.Test;
import org.junit.jupiter.api.Assertions;

public class NetworkConstructorTest {
    @Test
    public void constructorTest1(){
        Exception actual = Assertions.assertThrows(IncorrectDataException.class, () -> new Network(0, 1, null));
        IncorrectDataException expected = new IncorrectDataException("Network constructor - input layer size");
        Assertions.assertEquals(expected.getMessage(), actual.getMessage());
    }
    @Test
    public void constructorTest2(){
        Exception actual = Assertions.assertThrows(IncorrectDataException.class, () -> new Network(1, 0, null));
        IncorrectDataException expected = new IncorrectDataException("Network constructor - output layer size");
        Assertions.assertEquals(expected.getMessage(), actual.getMessage());
    }
    @Test
    public void constructorTest3(){
        Exception actual = Assertions.assertThrows(IncorrectDataException.class, () -> new Network(1, 1, new int[]{0}));
        IncorrectDataException expected = new IncorrectDataException("Network constructor - hidden layers sizes");
        Assertions.assertEquals(expected.getMessage(), actual.getMessage());
    }
    @Test
    public void constructorTest4(){
        try{ new Network(1,1,new int[] {1}) ;
        }catch (IncorrectDataException exception){
            Assertions.fail(exception.getLocalizedMessage());
        }
    }
}
