##  NSNPFuse

Codes for ***NSNPFuse: When Multi-focus Image Fusion Meets Nonlinear Spiking Neural P Systems.***

    sources\                               : MFIF datasets.
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

Since the size of **RealMFF dataset** is 800+MB, we can not upload it for exhibition. Download the RealMFF dataset from [GoogleDrive](https://drive.google.com/file/d/1UgV_AFmAlzZunaXmyVvoskbhbudr_SQp/view) or [BaiduYun](https://pan.baidu.com/s/13WfJ6kxEuaVvOla-OOsx0A) (Code：q3zz)
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

## 🏄Evaluation
We also provide evaluation metrics for testing fused images. If you want to obtain the fusion results in our paper, please run **main.m** directly from the Objective-evaluation-for-image-fusion folder.

![](https://github.com/MorvanLi/NSNPFuse-MFIF/blob/main/figs/evaluation.png)


You can modify the `fused_path` to test the results on different datasets. For example, to test on MFFW, set `fused_path = './results/MFFW/'`, or to test on MFI-WHU, set `fused_path = './results/MFI-WHU/'`.



## 🙌 Downstream Application Verification
**1. Salient Object Detectiont**

In the salient object detection task, we use [U2Net](https://github.com/xuebinqin/U-2-Net ) to evaluate the effectiveness of our method.

**2. Object Detectiont**

In the object detection task, we use [YOLOv5](https://github.com/ultralytics/yolov5) to evaluate the effectiveness of our method.

Note that the dataset used to perform target detection is [Road-MF](https://github.com/ixilai/SAMF), which requires multi-focus image fusion before object detection can be performed. **This dataset is in the sources folder.** In addition, the Road-MF dataset lacks category labels and therefore can only be evaluated qualitatively.