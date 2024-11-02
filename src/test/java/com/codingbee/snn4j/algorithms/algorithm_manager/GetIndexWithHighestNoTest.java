package com.codingbee.snn4j.algorithms.algorithm_manager;

import com.codingbee.snn4j.algorithms.Algorithms;
import com.codingbee.snn4j.exceptions.IncorrectDataException;
import org.junit.Test;
import org.junit.jupiter.api.Assertions;

public class GetIndexWithHighestNoTest {
    //Functionality tests:
    @Test
    public void getIndexWithHighestNoZest1(){
        int expected = 2;
        int actual = Algorithms.getIndexWithHighestVal(new double[]{1, 3, 8, 2 , 6});
        Assertions.assertEquals(expected, actual);
    }
    @Test
    public void getIndexWithHighestNoZest2(){
        int expected = 2;
        int actual = Algorithms.getIndexWithHighestVal(new double[]{1, 3, 8, 2 , 6});
        Assertions.assertEquals(expected, actual);
    }
    //Ege case: highest value at multiple indexes
    @Test
    public void getIndexWithHighestNoZest3(){
        int expected = 0;
        int actual = Algorithms.getIndexWithHighestVal(new double[]{12, 4, 9, 7 , 12});
        Assertions.assertEquals(expected, actual);
    }
    //Edge case: passed array is null
    @Test
    public void getIndexWithHighestNoZest4(){
        IncorrectDataException actual = Assertions.assertThrows(IncorrectDataException.class, () -> Algorithms.getIndexWithHighestVal(null));
        IncorrectDataException expected = new IncorrectDataException("Get index with highest value - the array must not be null");
        Assertions.assertEquals(expected.getLocalizedMessage(), actual.getLocalizedMessage());
    }
}
