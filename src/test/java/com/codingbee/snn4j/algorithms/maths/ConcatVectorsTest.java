package com.codingbee.snn4j.algorithms.maths;

import com.codingbee.snn4j.algorithms.Maths;
import com.codingbee.snn4j.exceptions.IncorrectDataException;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class ConcatVectorsTest {
    @Test
    public void nullTest(){
        float[] vector = new float[]{1, 2, 3};
        Assertions.assertThrows(IncorrectDataException.class, () -> Maths.concatVectors(null, vector));
    }

    @Test
    public void nullTest2(){
        float[] vector = new float[]{1, 2, 3};
        Assertions.assertThrows(IncorrectDataException.class, () -> Maths.concatVectors(vector, null));
    }

    @Test
    public void emptyArrayTest(){
        float[] vectorA = new float[]{1, 2, 3};
        float[] vectorB = new float[]{};

        float[] expected = new float[]{1, 2, 3};

        float[] actual = Maths.concatVectors(vectorA, vectorB);

        Assertions.assertArrayEquals(expected, actual);
    }

    @Test
    public void emptyArrayTest2(){
        float[] vectorA = new float[]{};
        float[] vectorB = new float[]{1, 2, 3};

        float[] expected = new float[]{1, 2, 3};

        float[] actual = Maths.concatVectors(vectorA, vectorB);

        Assertions.assertArrayEquals(expected, actual);
    }

    @Test
    public void emptyArrayTest3(){
        float[] vectorA = new float[]{};
        float[] vectorB = new float[]{};

        float[] expected = new float[]{};

        float[] actual = Maths.concatVectors(vectorA, vectorB);

        Assertions.assertArrayEquals(expected, actual);
    }

    @Test
    public void functionalityTest(){
        float[] vectorA = new float[]{1, 2, 3};
        float[] vectorB = new float[]{4, 5, 6};

        float[] expected = new float[]{1, 2, 3, 4, 5, 6};

        float[] actual = Maths.concatVectors(vectorA, vectorB);

        Assertions.assertArrayEquals(expected, actual);
    }

    @Test
    public void functionalityTest2(){
        float[] vectorA = new float[]{5.6f, 2.7f, 1.3f};
        float[] vectorB = new float[]{4.9f, 5.66f, 1.42f};

        float[] expected = new float[]{5.6f, 2.7f, 1.3f, 4.9f, 5.66f, 1.42f};

        float[] actual = Maths.concatVectors(vectorA, vectorB);

        Assertions.assertArrayEquals(expected, actual);
    }
}
