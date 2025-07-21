package com.codingbee.snn4j.algorithms.memory_utils;

import com.codingbee.snn4j.algorithms.MemoryUtils;
import com.codingbee.snn4j.exceptions.IncorrectDataException;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class ValidateMatrixTest {

    @Test
    void nullTest(){
        Assertions.assertThrows(IncorrectDataException.class, () -> MemoryUtils.validateMatrix(null));
    }

    @Test
    void jaggedArrayTest(){
        float[][] matrix = new float[][]{{1, 2, 3}, {1, 2, 3}, {1, 2}};
        Assertions.assertThrows(IncorrectDataException.class, () -> MemoryUtils.validateMatrix(matrix));
    }

    @Test
    void jaggedArrayTest2(){
        float[][] matrix = new float[][]{{1, 2, 3, 4}, {1, 2, 3}, {1, 2, 3}};
        Assertions.assertThrows(IncorrectDataException.class, () -> MemoryUtils.validateMatrix(matrix));
    }

    @Test
    void nullAsRowTest(){
        float[][] matrix = new float[][]{{1, 2, 3}, {1, 2, 3}, null};
        Assertions.assertThrows(IncorrectDataException.class, () -> MemoryUtils.validateMatrix(matrix));
    }

    @Test
    void validMatrixTest(){
        float[][] matrix = new float[][]{{1, 2, 3}, {1, 2, 3}, {1, 2, 3}};
        Assertions.assertDoesNotThrow(() -> MemoryUtils.validateMatrix(matrix));
    }

    @Test
    void validMatrixTest2(){
        float[][] matrix = new float[][]{{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}};
        Assertions.assertDoesNotThrow(() -> MemoryUtils.validateMatrix(matrix));
    }
}
