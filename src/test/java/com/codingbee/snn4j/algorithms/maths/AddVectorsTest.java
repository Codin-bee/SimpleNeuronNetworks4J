package com.codingbee.snn4j.algorithms.maths;

import com.codingbee.snn4j.algorithms.Maths;
import com.codingbee.snn4j.exceptions.IncorrectDataException;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class AddVectorsTest {

    @Test
    public void functionalityTest(){
        float[] vectorA = new float[]{1, 2.3f, 4, 5};
        float[] vectorB = new float[]{9.56f, 4.4f, 3, 1};

        float[] expected = new float[]{10.56f, 6.7f, 7, 6};
        float[] actual = Maths.addElementWise(vectorA, vectorB);

        Assertions.assertArrayEquals(expected, actual);
    }

    @Test
    public void functionalityTest2(){
        float[] vectorA = new float[]{76.45f, 573, 29.5f, 5};
        float[] vectorB = new float[]{12, 4.32f, 4, 3};

        float[] expected = new float[]{88.45f, 577.32f, 33.5f, 8};
        float[] actual = Maths.addElementWise(vectorA, vectorB);

        Assertions.assertArrayEquals(expected, actual);
    }

    @Test
    public void nullTest(){
        float[] vectorA = null;
        float[] vectorB = new float[]{2, 3, 4, 9.56f, 4.3f, 3, 1};

        Assertions.assertThrows(IncorrectDataException.class, () -> Maths.addElementWise(vectorA, vectorB));
    }

    @Test
    public void nullTest2(){
        float[] vectorA = new float[]{6, 8, 9, 2, 1, 2.3f, 4, 5};
        float[] vectorB = null;

        Assertions.assertThrows(IncorrectDataException.class, () -> Maths.addElementWise(vectorA, vectorB));
    }

    @Test
    public void differentLengthsTest(){
        float[] vectorA = new float[]{2, 1, 2.3f, 4, 5};
        float[] vectorB = new float[]{9.56f, 4.3f, 3, 1};

        Assertions.assertThrows(IncorrectDataException.class, () -> Maths.addElementWise(vectorA, vectorB));
    }
}
