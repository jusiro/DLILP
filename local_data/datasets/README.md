We suggest the following dataset organization to ease management and avoid modifying the source code.
The datasets structure looks like:

```
DLILP/
└── local_data/
    └── datasets/
        ├── CheXpert
        ├── COVID-19_Radiography_Dataset
        ├── MIMIC-CXR-2
        ├── NIH
        ├── PadChest
        ├── RSNA_PNEUMONIA
        ├── SIIM_Pneumothorax
        └── VinDr-PCXR
```

In the following, we provide specific download links and expected structure for each individual dataset.

### CheXpert

A large chest radiograph dataset with uncertainty labels extracted from radiology reports trough entity extraction methods.
Note that only labels (no text) are publicly available. You can find the dataset at: [LINK](https://stanfordmlgroup.github.io/competitions/chexpert/).


```
.
└── CheXpert/
    ├── CheXpert-v1.0/
    │   ├── train/
    │   │   ├── patientxxxx1/
    │   │   │   ├── study1/
    │   │   │   │   └── view_frontal.jpg
    │   │   │   └── ...
    │   │   ├── patientxxxx2
    │   │   └── ...
    │   └── valid/
    │       └── ...
    └── train_visualCheXbert.csv
    └── chexpert_5x200.csv
```

Note that `chexpert_5x200.csv` should be obtained from [GlorIA repository](https://stanfordmedicine.app.box.com/s/j5h7q99f3pfi7enc0dom73m4nsm6yzvh).

### MIMIC

A large chest radiograph dataset with radiology reports. In particular, we used the compressed images of the version
MIMIC-CXR-JPG v2.0.0. You can find the dataset at: [LINK](https://physionet.org/content/mimic-cxr-jpg/2.0.0/). The radiology
text reports are obtained from the raw MIMIC dataset , which can be found at: [LINK](https://physionet.org/content/mimic-cxr/2.0.0/).

```
.
└── MIMIC-CXR-2/
    └── 2.0.0/
        ├── files/
        │   ├── p10/
        │   │   ├── p10000032/
        │   │   │   ├── s50414267/
        │   │   │   │   ├── 02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg
        │   │   │   │   ├── 174413ec-4ec4c1f7-34ea26b7-c5f994f8-79ef1962.jpg
        │   │   │   │   ├── ...
        │   │   │   │   └── s50414267.txt
        │   │   │   └── ...
        │   │   └── ...
        │   └── ...
        ├── mimic-cxr-2.0.0-split.csv
        ├── mimic-cxr-2.0.0-metadata.csv
        └── labels_individual_sentences.csv
```

We created the file `labels_individual_sentences.csv` by extracting expert labels for the 10 target categories addressed in
CheXpert and MIMIC using entity extraction methods sentence-wise. You can refer to [chexpert-labeler](https://github.com/stanfordmlgroup/chexpert-labeler) 
where the tools for entity extraction methods are offered. The produced dataframe should follow this structure:

| patient   | sentences | No Finding | Enlarged Cardiomegaly | ...   | Support Devices |
|-----------|-----------|------------|-----------------------|-------|-----------------|
| s50414267 | "...."    | 1          |                       | ...   | 0               |
| s50414267 | "...."    | 0          | 1                     | ...   |                 |
| s50414267 | "...."    | 0          | 0                     | ...   |                 |
| s56699142 | "...."    |            |                       | ...   |                 |
| s56699142 | "...."    | 0          |                       | ...   | 1               |
| ...       | ...       | ...        | ...                   | ...   | ...             |

In particular, each row represents a sentence in the radiology report of one patient. Sentences are extracted combining sections
IMPRESSIONS and FINDINGS from the raw reports.

### PadChest

PadChest is a dataset containing Chest-X-ray images labeled using NLP-based entity extraction methods. It is worth mentioning that
a subset of PadChest is labeled directly by expert radiologists. You can find the dataset at: [LINK](http://bimcv.cipf.es/bimcv-projects/padchest).
Following MIMIC-CXR-JPG, we stored the images as '.jpeg' files to save disk space.

```
.
└── PadChest/
    ├── images/
    │   ├── ...
    │   ├── 99974151624878256478995523956634565424_f6ag9r.jpeg
    │   ├── 99976282796411202176162182849344921265_h61yy2.jpeg
    │   ├── 99994279947321985553707645848313304393_rk4e8v.jpeg
    │   ├── 99994279947321985553707645848313304393_syjs6f.jpeg
    │   └── ...
    └── PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv
```

### SSIM

SSIM-Pneumothorax is a dataset designed to classify (and if present, segment) pneumothorax from a set of chest radiographic
images. We employ it as a binary image-level classification task. You can find the dataset at: [LINK](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/).

```
.
└── SIIM_Pneumothorax/
    ├── dicom-images-test
    ├── dicom-images-train/
    │   ├── ...
    │   ├── 1.2.276.0.7230010.3.1.2.8323329.306.1517875162.312800/
    │   │   └── 1.2.276.0.7230010.3.1.3.8323329.306.1517875162.312799/
    │   │       └── 1.2.276.0.7230010.3.1.4.8323329.306.1517875162.312801.dcm
    │   ├── 1.2.276.0.7230010.3.1.2.8323329.301.1517875162.280318/
    │   │   └── ...
    │   └── ...
    └── train-rle.csv
```

### RNSA

RNSA dataset aims to detect visual signal for pneumonia in medical images. Specifically, your algorithm needs to automatically
locate lung opacities on chest radiographs. We used this dataset as an image-level binary classification dataset of pneumonia 
diseases, as recent literature has done. You can find the dataset at: [LINK](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/).

```
.
└── RSNA_PNEUMONIA/
    ├── stage_2_test_images/
    │   └── ...
    ├── stage_2_train_images/
    │   ├── ...
    │   ├── 0a0f91dc-6015-4342-b809-d19610854a21.dcm
    │   └── ...
    └── stage_2_detailed_class_info.csv
    └── stage_2_train_labels.csv
```

### COVID

This dataset is an assembly of different data sources which contain normal, covid, and two other lung findings (non-covid).
Although prior works only focus on binary classification COVID vs. Normal, we use the four available categories. You can 
find the dataset at: [LINK](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database).

```
.
└── COVID-19_Radiography_Dataset/
    ├── COVID/
    │   ├── images/
    │   │   ├── COVID-1.png
    │   │   ├── COVID-2.png
    │   │   └── ....
    │   └── masks/
    │       └── ...
    ├── Lung_Opacity/
    │   ├── images/
    │   │   ├── Lung_Opacity-1.png
    │   │   ├── Lung_Opacity-2.png
    │   │   └── ....
    │   └── masks/
    │       └── ...
    ├── Normal/
    │   ├── images/
    │   │   ├── Normal-1.png
    │   │   ├── Normal-2.png
    │   │   └── ....
    │   └── masks/
    │       └── ...
    └── ViralPneumonia/
        ├── images/
        │   ├── ViralPneumonia-1.png
        │   ├── ViralPneumonia-2.png
        │   └── ....
        └── masks/
            └── ...
```

### NIH

NIH dataset is also commonly known as ChestX-ray8/ChestX-ray14. This datasets contains different subsets of labeled conditions 
using entity extraction methods. You can download images at [LINK](hhttps://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345).
In this work, we used this dataset to explore the transferability of pre-trained models to generalize un novel diseases. Following this
objective, we employed the NIH-longtail partition, which contains 20 different labeled diseases. You can download the labels at 
[LINK](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/174256157515), which will provide the subfolder `./LongTailCXR/`.  

```
.
└── PadChest/
    ├── images/
    │   ├── ...
    │   ├── 99974151624878256478995523956634565424_f6ag9r.jpeg
    │   ├── 99976282796411202176162182849344921265_h61yy2.jpeg
    │   ├── 99994279947321985553707645848313304393_rk4e8v.jpeg
    │   ├── 99994279947321985553707645848313304393_syjs6f.jpeg
    │   └── ...
    └── PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv
```

### VinDR

VinDr-PCXR is a dataset containing frontal radiographs from pediatric patients, with up to 22 local lesions, and 6 diseases 
labeled by expert radiologists. You can download the labels at [LINK](https://physionet.org/content/vindr-cxr/1.0.0/).

```
.
└── VinDr-PCXR/
    ├── test/
    │   └── ...
    ├── train/
    │   ├── ...
    │   ├── fff630c5e8914944fd99321aa8336b01.dicom
    │   ├── ff75dc08847df23b0c0e488d0df97385.dicom
    │   └── ...
    ├── annotations_test.csv
    ├── annotations_train.csv
    ├── image_labels_test.csv
    └── image_labels_train.csv
```