package com.codingbee.snn4j.algorithms.memory_utils;

import com.codingbee.snn4j.algorithms.MemoryUtils;
import com.codingbee.snn4j.exceptions.IncorrectDataException;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class ArrayAllocation3DTest {

    @Test
    public void dimensionalityTest(){
        float[][][] original = new float[][][]{{{1, 2, 3}, {1, 2, 3, 4}, {1, 2, 3, 4, 5, 6, 7}}, {{1, 2}}};
        float[][][] copy = MemoryUtils.allocateArrayOfSameSize(original);

        Assertions.assertEquals(original.length, copy.length);

        for (int i = 0; i < original.length; i++) {
            Assertions.assertEquals(original[i].length, copy[i].length);
            for (int j = 0; j < original[i].length; j++) {
                Assertions.assertEquals(original[i][j].length, copy[i][j].length);
            }
        }
    }

    @Test
    public void dimensionalityTest2(){
        float[][][] original = new float[][][]{{{1, 2, 3, 4}, {1, 2, 3, 4, 5, 6, 7}}, {{1, 2}, {1, 2, 3, 4, 5}}};
        float[][][] copy = MemoryUtils.allocateArrayOfSameSize(original);

        Assertions.assertEquals(original.length, copy.length);

        for (int i = 0; i < original.length; i++) {
            Assertions.assertEquals(original[i].length, copy[i].length);
            for (int j = 0; j < original[i].length; j++) {
                Assertions.assertEquals(original[i][j].length, copy[i][j].length);
            }
        }
    }

    @Test
    public void nullTest(){
        Assertions.assertThrows(IncorrectDataException.class, () -> MemoryUtils.allocateArrayOfSameSize((float[][][]) null));
    }

    @Test
    public void nullElementTest(){
        float[][][] array = new float[][][]{{{1, 2, 3}, {1, 2, 3}}, {{1, 2, 3}, {1, 2, 3}, null}};
        Assertions.assertThrows(IncorrectDataException.class, () -> MemoryUtils.allocateArrayOfSameSize(array));
    }

    @Test
    public void nullElementTest2(){
        float[][][] array = new float[][][]{{{1, 2, 3}, {1, 2, 3}}, null};
        Assertions.assertThrows(IncorrectDataException.class, () -> MemoryUtils.allocateArrayOfSameSize(array));
    }
}
