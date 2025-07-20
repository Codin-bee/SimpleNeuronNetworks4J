package com.codingbee.snn4j.algorithms.maths;

import com.codingbee.snn4j.algorithms.Maths;
import com.codingbee.snn4j.exceptions.IncorrectDataException;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class GetIndexOfLargestElementTest {
    @Test
    public void nullTest(){
        Assertions.assertThrows(IncorrectDataException.class, () -> Maths.getIndexOfLargestElement(null));
    }

    @Test
    public void emptyArrayTest(){
        Assertions.assertThrows(IncorrectDataException.class, () -> Maths.getIndexOfLargestElement(new float[]{}));
    }

    @Test
    public void functionalityTest(){
        float[] array = new float[]{4.5f, 6.4f, 6.9f};
    }
}
