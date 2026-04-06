# Reproducing Published EEG MI Classification Results

Sources:
- [OpenBCI Public List](https://openbci.com/community/publicly-available-eeg-datasets)
- [Dataset 1: Left/Right Hand MI](https://gigadb.org/dataset/100295)
- [Dataset 4: The largest SCP data of Motor-Imagery](https://figshare.com/collections/A_large_electroencephalographic_motor_imagery_dataset_for_electroencephalographic_brain_computer_interfaces/3917698)
- [Dataset 8: High-Gamma Dataset](https://github.com/robintibor/high-gamma-dataset?tab=readme-ov-file)


### Data Summary

| Property           | DS1                               | DS4 HaLT                      | DS8      |
|--------------------|-----------------------------------|-------------------------------|----------|
| N subjects         | 50 (52 recorded, 2 excluded)      | 29 sessions (13 participants) | 14       |
| Total samples      | 10,120                            | 27,641                        |          |
| N channels         | 64                                | 21                            |          |
| Trial duration     | 2.0 s                             | 0.85 s                        | 4.5 s    |
| Sampling rate      | 512 Hz                            | 200 Hz                        | 500 Hz   |
| Sample size        | 65,536                            | 3,570                         |          |
| Frequency band     | 8–30 Hz                           |                               | 0-125 Hz |
| N classes          | 2                                 | 6                             | 4        |


### Evaluation Summary

| Metric                               | DS1                | DS4 HaLT   |
|--------------------------------------|--------------------|------------|
| Mean Accuracy Per Subject (Baseline) | 67.46% ± 13.17%    | 57% ± 20%  |
| Discriminative Subjects (Baseline)   | 76% (38/50)        | N/A        |
