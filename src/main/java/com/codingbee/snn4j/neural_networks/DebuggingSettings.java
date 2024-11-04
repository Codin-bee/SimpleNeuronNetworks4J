package com.codingbee.snn4j.neural_networks;

public class DebuggingSettings {
    private boolean startEndPrint;
    private boolean everyIterationPrint;
    private boolean saveCostValues;
    private String costSavingFilePath;

    @SuppressWarnings("unused")
    public DebuggingSettings(boolean startEndPrint, boolean everyIterationPrint, boolean saveCostValues, String costSavingFilePath) {
        this.startEndPrint = startEndPrint;
        this.everyIterationPrint = everyIterationPrint;
        this.saveCostValues = saveCostValues;
        this.costSavingFilePath = costSavingFilePath;
    }

    public DebuggingSettings(){
        this(false, false, false, null);
    }

    @SuppressWarnings("unused")
    public boolean isStartEndPrint() {
        return startEndPrint;
    }

    @SuppressWarnings("unused")
    public void setStartEndPrint(boolean startEndPrint) {
        this.startEndPrint = startEndPrint;
    }

    @SuppressWarnings("unused")
    public boolean isEveryIterationPrint() {
        return everyIterationPrint;
    }

    @SuppressWarnings("unused")
    public void setEveryIterationPrint(boolean everyIterationPrint) {
        this.everyIterationPrint = everyIterationPrint;
    }

    @SuppressWarnings("unused")
    public boolean isSaveCostValues() {
        return saveCostValues;
    }

    @SuppressWarnings("unused")
    public void setSaveCostValues(boolean saveCostValues) {
        this.saveCostValues = saveCostValues;
    }

    @SuppressWarnings("unused")
    public String getCostSavingFilePath() {
        return costSavingFilePath;
    }

    @SuppressWarnings("unused")
    public void setCostSavingFilePath(String costSavingFilePath) {
        this.costSavingFilePath = costSavingFilePath;
    }
}
