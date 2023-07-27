# SemanticRT
This repository provides the **SemanticRT dataset** and **ECM code** for multispectral semantic segmentation (MSS). The repository is structured as follows.

1. [Task Introduction](#overall)
2. [SemanticRT Dataset](#SemanticRT)
3. [ECM Source Code](#ECM)

------

<a name="overall"></a>   
## Task Introduction 

![avatar](https://github.com/jiwei0921/SemanticRT/blob/main/intro.png)  

**Introduction Figure**. Visual illustration of the advantages of employing multispectral (RGB-Thermal) images for semantic segmentation. The complementary nature of RGB and thermal images are highlighted using yellow and green boxes, respectively. The RGB-only method, DeepLabV3+, is susceptible to incorrect segmentation or even missing target objects entirely. In contrast, multispectral segmentation methods, e.g., EGFNet and our ECM method, which incorporate thermal infrared information, effectively identify the segments within the context. Particularly, our results are visually closer to the ground truths compared to the state-of-the-art EGFNet.

------

<a name="SemanticRT"></a> 
## SemanticRT Dataset

SemanticRT dataset - the largest MSS dataset to date, comprises 11,371 high-quality, pixel-level annotated RGB-thermal image pairs. It covers a wide range of challenging scenarios in adverse lighting conditions such as low-light and pitch black, as displayed in the figure below.

![avatar](https://github.com/jiwei0921/SemanticRT/blob/main/dataset.png)

### Getting Started

+ **Dataset Access**

Download the SemanticRT dataset (Google Drive), which is structured as follows:

```
SemanticRT_dataset/
├─ train.txt
├─ val.txt
├─ test.txt
├─ test_day.txt
├─ test_night.txt
├─ test_mo.txt
├─ test_xxx.txt
│ ···
├─ rgb/
│  ├─ ···
├─ thermal/
│  ├─ ···
├─ labels/
│  ├─ ···
···
```
Training/testing/validation splits can be found in `train.txt`, `test_xxx.txt` or `val.txt`.

+ **Dataset ColorMap**

Here is the reference for SemanticRT dataset color visualization.
```
[
    (0, 0, 0),          # 0: background (unlabeled)
    (72, 61, 39),       # 1: car stop
    (0, 0, 255),        # 2: bike
    (148, 0, 211),      # 3: bicyclist
    (128, 128, 0),      # 4: motorcycle
    (64, 64, 128),      # 5: motorcyclist
    (0, 139, 139),      # 6: car
    (131, 139, 139),    # 7: tricycle
    (192, 64, 0),       # 8: traffic light
    (126, 192, 238),    # 9: box
    (244, 164, 96),     # 10:pole
    (211, 211, 211),    # 11:curve
    (205, 155, 155),    # 12:person
]
```

+ **Dataset Acknowledgement** 

Our SemanticRT dataset is mainly based on LLVIP as well as other RGBT sources (OSU and INO). They are annotated and adjusted to better fit the MSS task. All data and annotations provided are strictly intended for ***non-commercial research purpose only***. If you are interested in our SemanticRT dataset, we sincerely appreciate your citation of our work and strongly encourage you to cite the source datasets mentioned above.

------

<a name="ECM"></a> 
## ECM Source Code


Coming Soon!

The authors promise that the source code and SemanticRT dataset will be made publicly available to the research community upon paper acceptance. It is currently available from the corresponding author (wji3@ualberta.ca) upon reasonable request. 

+ **Code Acknowledgement** 

This code repository was originally built from [EGFNet](https://github.com/shaohuadong2021/egfnet). It was modified and extended to support our multispectral segmentation setting.

------

## Citation

```
@InProceedings{ji2023semanticrt,
      title     = {SemanticRT: A Large-Scale Dataset and Method for Robust Semantic Segmentation in Multispectral Images},
      author    = {Ji, Wei and Li, Jingjing and Bian, Cheng and Zhang, Zhicheng and Cheng, Li},
      booktitle = {Proceedings of the 31th ACM International Conference on Multimedia},
      month     = {October},
      year      = {2023}
}
```

If you have any further questions, please email us at wji3@ualberta.ca.
