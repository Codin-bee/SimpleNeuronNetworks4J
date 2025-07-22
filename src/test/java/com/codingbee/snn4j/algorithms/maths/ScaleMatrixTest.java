package com.codingbee.snn4j.algorithms.maths;

import com.codingbee.snn4j.algorithms.Maths;
import com.codingbee.snn4j.exceptions.IncorrectDataException;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class ScaleMatrixTest {

    @Test
    public void functionalityTest(){
        float[][] expected = new float[][]{{2, 4, 6}, {2, 4, 6}, {2, 4, 6}};

        float[][] actual = new float[][]{{1, 2, 3}, {1, 2, 3}, {1, 2, 3}};
        Maths.scale(actual, 2);

        for (int i = 0; i < expected.length; i++) {
            Assertions.assertArrayEquals(expected[i], actual[i], 0.000001f);
        }
    }

    @Test
    public void functionalityTest2(){
        float[][] expected = new float[][]{{5, 6, 8}, {3, 4, 55}, {2, 4, 6}};

        float[][] actual = new float[][]{{10, 12, 16}, {6, 8, 110}, {4, 8, 12}};
        Maths.scale(actual, 0.5f);

        for (int i = 0; i < expected.length; i++) {
            Assertions.assertArrayEquals(expected[i], actual[i], 0.000001f);
        }
    }

    @Test
    public void invalidMatrixTest(){
        float[][] matrix = new float[][]{{1, 2, 3}, {1, 2, 3, 4}, {1, 2, 3}};
        Assertions.assertThrows(IncorrectDataException.class, () -> Maths.scale(matrix, 2));
    }

    @Test
    public void nullTest(){
        Assertions.assertThrows(IncorrectDataException.class, () -> Maths.scale(null, 2));
    }

    @Test
    public void nullElementTest(){
        float[][] matrix = new float[][]{{1, 2, 3}, null, {1, 2, 3}};
        Assertions.assertThrows(IncorrectDataException.class, () -> Maths.scale(matrix, 2));
    }
}
