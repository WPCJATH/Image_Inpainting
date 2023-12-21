# Course Project of IEMS5726 Data Science in Practice (Fall 2023)
Here is the course project files of WANG XIANGZHI. The topic is Image Inpainting under Deep Image Processing. This specific project provides full pipeline of both non-deep and deep learning methods on image inpainting tasks.

All references are well-citated, and all codes are well documented and commented. This project is under [MIT License](#license), and doesn't disclose any sensitive and copyright-protected information about the course and lecturer.

**Date of Creation:** Dec-16-2023   
**Github URL:** https://github.com/WPCJATH/Image_Inpainting   
<br>
\* The  repository is made publicly available after the project DUE DATE: Dec-21-2023

## Author Information
- NAME: WANG Xiangzhi
- STUDENT ID: 1155200446
- E-MAIL: 1155200446@link.cuhk.edu.hk

## Installation
This project is compeletly runnable on a windows machine with Nvidia GPU and conda environment, other
platform or environment is not tested.

To install the conda environment with GPU support, run the commands below:
```batch
conda env create --file environment.yml
conda activate inpainting
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
or execute the batch file `env_install_conda.bat`

## Web Demo
Run the command below:
```batch
conda activate inpainting
python app.py
```
Or execute the batch file `start_service.bat`   

## Train
The training config is at `train-place.yaml` with commented instructions   

**Pre-trained model URL:** https://drive.google.com/u/0/uc?id=1L63oBNVgz7xSb_3hGbUdkYW1IuRgMkCa&export=download   
**Please put the pre-trained model under folder `model/pretrained`**


**Retrained model URL:** https://drive.google.com/drive/folders/1d3PVcqljGTz_Z8Dc1J1aP-KdIsLvrtbR?usp=sharing    
**Please put the retrained model under folder `model/retrained` and specify the retrain model name in the config file (`states_25000.pth` is the default)**

**Data URL:** The same as the retrained model
**Please put the unzipped data folder under folder `data/`, so the data can be accessed by path `data/snowfield`**

Run the below command:
```batch
conda activate inpainting
python train_model.py
```

## Visualization and evaluation
Run the below command:
```batch
conda activate inpainting
python data_visiualization.py
```

## Training logging
Download the training log at the same url as the retrained model, and put the log file under path `tb_logs/places/model_retrain`  
Run the following command:
```
conda activate inpainting
tensorboard --logdir=tb_logs\places2\model_retrain --host localhost --port 8088
```
Or execute the batch file `tensorboard_log.bat`

Then, the log can be viewed under http://localhost:8088   


## License
>The MIT License (MIT)
>
>Copyright (c) 2023 Eric Wang
>
>Permission is hereby granted, free of charge, to any person obtaining a copy
>of this software and associated documentation files (the "Software"), to deal
>in the Software without restriction, including without limitation the rights
>to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
>copies of the Software, and to permit persons to whom the Software is
>furnished to do so, subject to the following conditions:
>
>The above copyright notice and this permission notice shall be included in all
>copies or substantial portions of the Software.
>
>THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
>IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
>FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
>AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
>LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
>OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
>SOFTWARE.
>

