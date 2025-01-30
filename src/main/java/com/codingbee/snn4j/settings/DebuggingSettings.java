package com.codingbee.snn4j.settings;

@SuppressWarnings("unused")
public class DebuggingSettings {
    private boolean startEndPrint;
    private boolean everyIterationPrint;
    private boolean saveCostValues;
    private boolean savePeriodically;
    private int periodicSavingInterval;
    private String costSavingFilePath;

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

    //region GETTERS

    public boolean isStartEndPrint() {
        return startEndPrint;
    }

    public boolean isEveryIterationPrint() {
        return everyIterationPrint;
    }

    public boolean isSaveCostValues() {
        return saveCostValues;
    }

    public boolean isSavePeriodically() {
        return savePeriodically;
    }

    public int getPeriodicSavingInterval() {
        return periodicSavingInterval;
    }

    public String getCostSavingFilePath() {
        return costSavingFilePath;
    }

    //endregion
}
