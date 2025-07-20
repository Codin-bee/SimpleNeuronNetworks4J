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
        int expected = 2;
        int actual = Maths.getIndexOfLargestElement(array);

        Assertions.assertEquals(expected, actual);
    }

    @Test
    public void functionalityTest2(){
        float[] array = new float[]{-4.8f, -3.573f, 68.9f, 0f, 4.802f, 982.4370f};
        int expected = 5;
        int actual = Maths.getIndexOfLargestElement(array);

        Assertions.assertEquals(expected, actual);
    }

    @Test
    public void repeatingLargestElementTest(){
        float[] array = new float[]{0.849f, -4.6f, -5.6f, 23f, -5.98f, 23f, 0f};
        int expected = 3;
        int actual = Maths.getIndexOfLargestElement(array);

        Assertions.assertEquals(expected, actual);
    }

    @Test
    public void repeatingLargestElementTest2(){
        float[] array = new float[]{-4, -83, -45, -64, -3, -83, -99, -5, -3};
        int expected = 4;
        int actual = Maths.getIndexOfLargestElement(array);

        Assertions.assertEquals(expected, actual);
    }
}
