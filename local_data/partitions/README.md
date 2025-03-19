Once the datasets have been downloaded and structured following [`../datasets/README.md`](../datasets/README.md),
follow the instructions indicated here to prepare pre-training and transferability partitions.

The expected outcome should look like:

```
.
└── DLILP/
    └── local_data/
        └── partitions/
            ├── pretraining/
            │   ├── CheXpert-train-frontal.csv
            │   ├── MIMIC-CXR-2-train-frontal.csv
            │   └── PadChest-train-frontal.csv
            └── transferability/
                ├── chexpert_5x200.csv
                ├── covid_test.csv
                ├── covid_train.csv
                ├── mimic_5x200.csv
                ├── nih_train.csv
                ├── nih_test.csv
                ├── rsna_pneumonia_test.csv
                ├── rsna_pneumonia_train.csv
                ├── SIIM_test.csv
                ├── SIIM_train.csv
                ├── vindr_test.csv
                └── vindr_train.csv
```

### Instructions

1. Store`chexpert_5x200.csv` from [GlorIA repository](https://stanfordmedicine.app.box.com/s/j5h7q99f3pfi7enc0dom73m4nsm6yzvh) at `./local_data/partitions/transferability/`. 
2. You can run the following code if you have properly set all datasets. Otherwise, you can inspect [`partitions.py`](partitions.py) 
and run individual codes for each dataset.

```
python partitions.py
```