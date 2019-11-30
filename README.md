# Sub-pixel Detection

## 1. Project description

Sub-pixel detection project in SJTU CS386. Our object is to detect interest points in sub-pixel accuracy. We proposed SPResNet, using residual blocks, upsampling and pixel shuffle to extract sub-pixel points.

The implementation is based on Pytorch. Experiments runs on MacOS.

## 2. Usage

### 2.1. install requirements

```bash
pip install -r requirements.txt
```

### 2.2. prepare datasets

Put train HR(High-Resolution) images in `./data/intput/train_input`, put validation HR images in `./data/input/valid_input`, and (optionally) put testing HR images in `./data/input/test_input`.   We already put some demo images in these folders.

### 2.3. train and test

The interface is `main.py`. The following command will run the program by default settings. Defaultly it will pre-process the data, then train and test.

```bash
python main.py
```

Check `./utils/option.py` for arguments details. Or running help:

```bash
python main.py -h
```

### 2.4. Output

Output of test is in `./data/output/test_output`.

## 3.remarks

### 3.1. Dataset download

We downloaded DIV2K dataset from https://github.com/xinntao/BasicSR/wiki/Prepare-datasets-in-LMDB-format .  Other datasets are also accpetable.

### 3.2. Image type

By default, only `.png` images are valid input. If other type of images need to be supported, please modify the following code in `./utils/dataPrepare.py` at Line14.

```
if name.endswith(".png"): 
```

