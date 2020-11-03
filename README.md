# Self-Challenging Improves Cross-Domain Generalization
This is the official implementation of: 

Zeyi Huang*, Haohan Wang*, Eric P. Xing, and Dong Huang, Self-Challenging Improves Cross-Domain Generalization, ECCV, 2020 (Oral), [arxiv version](https://arxiv.org/abs/2007.02454).

Update: To mitigate fluctuation in different environments, we modify RSC in an curriculum manner. Also, we unify RSC for different network architectures. If you have any questions about the code, feel free to contact me or pull a issue.

### Citation: 

```bash
@inproceedings{huangRSC2020,
  title={Self-Challenging Improves Cross-Domain Generalization},
  author={Zeyi Huang and Haohan Wang and Eric P. Xing and Dong Huang},
  booktitle={ECCV},
  year={2020}
}
```

## Installation

### Requirements:

- Python ==3.7
- Pytorch ==1.1.0
- Torchvision == 0.3.0
- Cuda ==10.0
- Tensorflow ==1.14
- GPU: RTX 2080

## Data Preparation
Download PACS dataset from [here](http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017). Once you have download the data, you must update the files in data/correct_txt_list to match the actual location of your files. Note: make sure you use the same train/val/test split in PACS paper.

## Runing on PACS dataset
Experiments with different source/target domains are listed in train.py(L145-152).
To train a ResNet18, simply run:
```bash
  python train.py --net resnet18
```


## Other pretrained models
New ImageNet ResNet baselines training by RSC.

| Backbone        | Top-1 Acc % |Top-5 Acc % | pth models |
| :--------------:| :--------------: | :------------:  |:------------:  |
| ResNet-50       |77.18           |93.53            |[download](https://cmu.box.com/s/wpcy4mwkfm7gku3q4b115d5y1t69i4s4)   |
| ResNet-101      |78.23           |94.16            |[download](https://cmu.box.com/s/wpcy4mwkfm7gku3q4b115d5y1t69i4s4)   |
| ResNet-152      |78.89           |94.43            | download   |


## Acknowledgement
We borrowed code and data augmentation techniques from [Jigen](https://github.com/fmcarlucci/JigenDG).
