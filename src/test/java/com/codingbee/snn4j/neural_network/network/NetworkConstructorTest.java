package com.codingbee.snn4j.neural_network.network;

import com.codingbee.snn4j.exceptions.IncorrectDataException;
import com.codingbee.snn4j.neural_network.Network;
import org.junit.Test;
import org.junit.jupiter.api.Assertions;

public class NetworkConstructorTest {
    @Test
    public void constructorTest1(){
        Exception actual = Assertions.assertThrows(IncorrectDataException.class, () -> new Network(0, 1, null, 0));
        IncorrectDataException expected = new IncorrectDataException("Network constructor - input layer size");
        Assertions.assertEquals(expected.getMessage(), actual.getMessage());
    }
    @Test
    public void constructorTest2(){
        Exception actual = Assertions.assertThrows(IncorrectDataException.class, () -> new Network(1, 0, null, 0));
        IncorrectDataException expected = new IncorrectDataException("Network constructor - output layer size");
        Assertions.assertEquals(expected.getMessage(), actual.getMessage());
    }
    @Test
    public void constructorTest3(){
        Exception actual = Assertions.assertThrows(IncorrectDataException.class, () -> new Network(1, 1, new int[]{0}, 0));
        IncorrectDataException expected = new IncorrectDataException("Network constructor - hidden layers sizes");
        Assertions.assertEquals(expected.getMessage(), actual.getMessage());
    }
    @Test
    public void constructorTest4(){
        try{ new Network(1,1,new int[] {1}, 0) ;
        }catch (IncorrectDataException exception){
            Assertions.fail(exception.getLocalizedMessage());
        }
    }
}
