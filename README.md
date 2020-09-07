# Self-Challenging Improves Cross-Domain Generalization (In progress)
This is the official implementation of: 

Zeyi Huang, Haohan Wang, Eric P. Xing, and Dong Huang, Self-Challenging Improves Cross-Domain Generalization, ECCV, 2020 (Oral), [arxiv version](https://arxiv.org/abs/2007.02454).

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

- Python >=3.7
- Pytorch>=1.0

Download PACS dataset from [here](http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017). Once you have download the data, you must update the files in data/txt_list to match the actual location of your files.

### Step-by-step installation


## Training on PACS dataset
Experiments with different source/target domains are listed in train.py(L151-158).

## Testing on PACS dataset


## Other pretrained models
New ImageNet ResNet baselines training by RSC.

| Backbone        | Top-1 Acc % |Top-5 Acc % | pth models |
| :--------------:| :--------------: | :------------:  |:------------:  |
| ResNet-50       |77.18           |93.53            |[download](https://cmu.box.com/s/wpcy4mwkfm7gku3q4b115d5y1t69i4s4)   |
| ResNet-101      |78.23           |94.16            | download   |
| ResNet-152      |78.89           |94.43            | download   |
