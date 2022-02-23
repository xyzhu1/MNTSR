# MNTSR
This is the official code of the paper "Scene Text Image Super-Resolution with Self-Supervised Memory Learning"

## Requirements
- easydict==1.9
- editdistance==0.5.3
- lmdb==1.2.1
- matplotlib==3.3.4
- numpy
- opencv-python==4.5.2.52
- Pillow==8.2.0
- six
- tensorboard==2.5.0
- tensorboard-data-server==0.6.1
- tensorboard-plugin-wit==1.8.0
- torch==1.2.0
- torchvision==0.2.1
- tqdm==4.61.0
- pyyaml
- ipython
- future

## Dataset
Download all resources at [BaiduYunDisk](https://pan.baidu.com/s/1P_SCcQG74fiQfTnfidpHEw) with password: stt6, or [Dropbox](https://www.dropbox.com/sh/f294n405ngbnujn/AABUO6rv_5H5MvIvCblcf-aKa?dl=0)

* TextZoom dataset
* Pretrained weights of CRNN 
* Pretrained weights of Transformer-based recognizer

All the resources shoulded be placed under ```./dataset/mydata```, for example
```python
./dataset/mydata/train1
./dataset/mydata/train2
./dataset/mydata/pretrain_transformer.pth
...
```


## Training
Please remember to modify the experiment name. Two text-focused modules are activated whenever ```--text_focus``` is used
```python
CUDA_VISIBLE_DEVICES=GPU_NUM python main.py --batch_size=16 --STN --exp_name EXP_NAME --text_focus
```

## Testing
```python
CUDA_VISIBLE_DEVICES=GPU_NUM python main.py --batch_size=16 --STN --exp_name EXP_NAME --text_focus --resume YOUR_MODEL --test --test_data_dir ./dataset/mydata/test
```
