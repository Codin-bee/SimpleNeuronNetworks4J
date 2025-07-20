package com.codingbee.snn4j.algorithms.memory_utils;

import com.codingbee.snn4j.algorithms.MemoryUtils;
import com.codingbee.snn4j.exceptions.IncorrectDataException;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class ArrayAllocation2DTest {

    @Test
    public void arrayDimensionalityTest1(){
        float[][] original = new float[][]{{1, 2, 3}, {1, 2, 3, 4}, {1, 2, 3, 4, 5, 6, 7}};
        float[][] copy = MemoryUtils.allocateArrayOfSameSize(original);

        Assertions.assertEquals(original.length, copy.length);

        for (int i = 0; i < original.length; i++) {
            Assertions.assertEquals(original[i].length, copy[i].length);
        }
    }

    @Test
    public void arrayDimensionalityTest2(){
        float[][] original = new float[][]{{1, 2, 3, 4, 5, 6}, {1}, {}};
        float[][] copy = MemoryUtils.allocateArrayOfSameSize(original);

        Assertions.assertEquals(original.length, copy.length);

        for (int i = 0; i < original.length; i++) {
            Assertions.assertEquals(original[i].length, copy[i].length);
        }
    }

    @Test public void nullTest(){
        Assertions.assertThrows(IncorrectDataException.class, () -> MemoryUtils.allocateArrayOfSameSize((float[][]) null));
    }
}
