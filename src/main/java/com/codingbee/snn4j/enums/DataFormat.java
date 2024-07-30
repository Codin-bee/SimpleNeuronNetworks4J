package com.codingbee.snn4j.enums;

import com.codingbee.snn4j.objects_for_parsing.JsonOne;
import com.codingbee.snn4j.objects_for_parsing.JsonTwo;

public enum DataFormat {
    /**Json containing one number, which is the correct output index(starting at 0), and array of values inserted into input layer,
     *  which should give off that correct index. Corresponds to this {@link JsonOne object}.*/
    JSON_ONE,
    /**Json containing two arrays. First one contains values inserted into input layer, second values expected in the output layer after processing them.
     * Corresponds to this {@link JsonTwo object}.*/
    JSON_TWO,
}
