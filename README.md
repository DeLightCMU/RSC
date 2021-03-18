# Self-Challenging Improves Cross-Domain Generalization

This is the official implementation of: 

**Zeyi Huang', Haohan Wang', Eric P. Xing, and Dong Huang**, ***Self-Challenging Improves Cross-Domain Generalization***, **ECCV, 2020 (Oral)**, [arxiv version](https://arxiv.org/abs/2007.02454).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/self-challenging-improves-cross-domain/domain-generalization-on-office-home)](https://paperswithcode.com/sota/domain-generalization-on-office-home?p=self-challenging-improves-cross-domain)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/self-challenging-improves-cross-domain/domain-generalization-on-pacs-2)](https://paperswithcode.com/sota/domain-generalization-on-pacs-2?p=self-challenging-improves-cross-domain)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/self-challenging-improves-cross-domain/domain-generalization-on-vlcs)](https://paperswithcode.com/sota/domain-generalization-on-vlcs?p=self-challenging-improves-cross-domain)

**Notice** about DG task: In order to get the same results in the testing part, you should use the same environment configuration [here](https://github.com/DeLightCMU/RSC/blob/master/Domain_Generalization/env.txt), including software, hardware and random seed. When using a different environment configuration, similar to other DG repositories, you need to tune the parameters a little bit. According to my observations, a simple larger batch size and early stop can solve the problem. If you still can't solve the problem, don't panic! send me an email(zeyih(at)andrew(dot)cmu(dot)edu) with your environment. I'll help you out.

**Update**: To mitigate fluctuation in different environments, we modify RSC in a curriculum manner. Also, we unify RSC for different network architectures. If you have any questions about the code, feel free to contact me or pull a issue.



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
Download PACS dataset from [here](http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017). Once you have download the data, you must update the files in data/correct_txt_list to match the actual location of your files. **Note**: make sure you use the same train/val/test split in PACS paper.

## Runing on PACS dataset
Experiments with different source/target domains are listed in train.py(L145-152).

To train a ResNet18, simply run:
```bash
  python train.py --net resnet18
```

To test a ResNet18, you can download RSC model below and [logs](https://cmu.box.com/s/yvymx574mr9u76lhqfa01rynimy9tv1p):
| Backbone        | Target Domain |Acc %            | models |
| :--------------:| :-----------: | :------------:  |:------------: |
| ResNet-18       |Photo          |96.05            |[download](https://cmu.box.com/s/hma6aw2ubcjyxpczhto6zortwf8ufin6)   |
| ResNet-18       |Sketch         |82.67            |[download](https://cmu.box.com/s/hfhgwsciz2a6aeg8jhffgwt5yh3dvenq)   |
| ResNet-18       |Cartoon        |81.61            |[download](https://cmu.box.com/s/9rw7z2gxdlq9fsa5sfamjfj1xwj95d54)   |
| ResNet-18       |Art            |85.16            |[download](https://cmu.box.com/s/ixfrzmanpv9t0koutiuaax91a26ylgit)  |


## To Do
Faster-RCNN

## Other pretrained models
New ImageNet ResNet baselines training by RSC.

| Backbone        | Top-1 Acc % |Top-5 Acc % | pth models |
| :--------------:| :--------------: | :------------:  |:------------:  |
| ResNet-50       |77.18           |93.53            |[download](https://cmu.box.com/s/wpcy4mwkfm7gku3q4b115d5y1t69i4s4)   |
| ResNet-101      |78.23           |94.16            |[download](https://cmu.box.com/s/wpcy4mwkfm7gku3q4b115d5y1t69i4s4)   |


## Acknowledgement
We borrowed code and data augmentation techniques from [Jigen](https://github.com/fmcarlucci/JigenDG), [ImageNet-pytorch](https://github.com/pytorch/examples/tree/master/imagenet).
