package com.codingbee.snn4j.neural_networks;

public class DebuggingSettings {
    protected boolean startEndPrint;
    protected boolean everyIterationPrint;
    protected boolean saveCostValues;
    protected boolean savePeriodically;
    protected int periodicSavingInterval;
    protected String costSavingFilePath;

    public DebuggingSettings(boolean startEndPrint, boolean everyIterationPrint, boolean saveCostValues, boolean savePeriodically, int periodicSavingInterval, String costSavingFilePath) {
        this.startEndPrint = startEndPrint;
        this.everyIterationPrint = everyIterationPrint;
        this.saveCostValues = saveCostValues;
        this.savePeriodically = savePeriodically;
        this.periodicSavingInterval = periodicSavingInterval;
        this.costSavingFilePath = costSavingFilePath;
    }

    public DebuggingSettings(){
        this(false, false, false, false, 1, null);
    }

    public void setStartEndPrint(boolean startEndPrint) {
        this.startEndPrint = startEndPrint;
    }

    public void setEveryIterationPrint(boolean everyIterationPrint) {
        this.everyIterationPrint = everyIterationPrint;
    }

    public void setSaveCostValues(boolean saveCostValues) {
        this.saveCostValues = saveCostValues;
    }

    public void setSavePeriodically(boolean savePeriodically) {
        this.savePeriodically = savePeriodically;
    }

    public void setPeriodicSavingInterval(int periodicSavingInterval) {
        this.periodicSavingInterval = periodicSavingInterval;
    }

    public void setCostSavingFilePath(String costSavingFilePath) {
        this.costSavingFilePath = costSavingFilePath;
    }
}
