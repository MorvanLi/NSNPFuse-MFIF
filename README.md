##  NSNPFuse

Codes for ***NSNPFuse: When Multi-focus Image Fusion Meets Nonlinear Spiking Neural P Systems.***

    results\                               : All fused images for the three MFIF datasets.
    Objective-evaluation-for-image-fusion\ : Code for evaluation metrics
    weights                                : Model weight
    train.py                               : Code of training
    test.py                                : Code of test

## 🌐 Usage
### 🏊 Training
**1. Virtual Environment**

```
# create virtual environment
conda create -n nsnpfuse python=3.8.10
conda activate nsnpfuse
# select pytorch version yourself
# install nsnpfuse requirements
pip install -r requirements.txt
```
**2. Data Preparation**
Unfortunately, since the size of **RealMFF dataset** is 800+MB, we can not upload it for exhibition. Download the RealMFF dataset from [GoogleDrive](https://drive.google.com/file/d/1UgV_AFmAlzZunaXmyVvoskbhbudr_SQp/view) or [BaiduYun](https://pan.baidu.com/s/13WfJ6kxEuaVvOla-OOsx0A) (Code：q3zz)
and place it in the folder ``'./TrainSet/'``. 

**3. NSNPFuse Training**

Run 
```
python train.py
```
and the trained model is available in ``'./weights/'``.

**3. NSNPFuse Testing**
The test datasets used in the paper have been stored in ``'./sources/Lytro'`` for Lytro, ``'./test_img/MFFW'`` for MFFW, ``'./sources/MFI-WHU'``  for MFI-WHU. 

**All fused images for the three MFIF datasets in the results folder .**

```
python test.py
```

## Evaluation Metrics
We also provide evaluation metrics for testing fused images. If you want to obtain the fusion results in our paper, please run **main.m** directly from the Objective-evaluation-for-image-fusion folder.

![](https://github.com/MorvanLi/NSNPFuse-MFIF/blob/main/figs/evaluation.png)


You can modify the `fused_path` to test the results on different datasets. For example, to test on MFFW, set `fused_path = './results/MFFW/'`, or to test on MFI-WHU, set `fused_path = './results/MFI-WHU/'`.

