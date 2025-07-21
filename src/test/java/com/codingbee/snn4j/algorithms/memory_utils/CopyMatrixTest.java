package com.codingbee.snn4j.algorithms.memory_utils;

import com.codingbee.snn4j.algorithms.MemoryUtils;
import com.codingbee.snn4j.exceptions.IncorrectDataException;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class CopyMatrixTest {

    @Test
    public void functionalityTest(){
        float[][] matrix = new float[][]{{1, 2, 3}, {1, 2, 3}, {1, 2, 3}};
        float[][] copy = MemoryUtils.copyMatrix(matrix);

        for (int i = 0; i < matrix.length; i++) {
            Assertions.assertArrayEquals(matrix[i], copy[i]);
        }
    }

    @Test
    public void functionalityTest2(){
        float[][] matrix = new float[][]{{1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}};
        float[][] copy = MemoryUtils.copyMatrix(matrix);

        for (int i = 0; i < matrix.length; i++) {
            Assertions.assertArrayEquals(matrix[i], copy[i]);
        }
    }

    @Test
    public void invalidMatrixTest(){
        float[][] matrix = new float[][]{{1, 2, 3}, {1, 2, 3}, {1, 2, 3, 4}};
        Assertions.assertThrows(IncorrectDataException.class, () -> MemoryUtils.copyMatrix(matrix));
    }

    @Test
    public void nullElementTest(){
        float[][] matrix = new float[][]{{1, 2, 3}, null};
        Assertions.assertThrows(IncorrectDataException.class, () -> MemoryUtils.copyMatrix(matrix));
    }

    @Test
    public void nullTest(){
        Assertions.assertThrows(IncorrectDataException.class, () -> MemoryUtils.copyMatrix(null));
    }
}
