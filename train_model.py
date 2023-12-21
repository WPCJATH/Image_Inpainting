'''
The python script for data training of this specific inpainting task.

Author: WANG Xiangzhi
Date: 21-Dec-2023
'''

import data_analysis2

mode = "train" # "train" or "infer_visual"
data_path = "data/snowfield"

if __name__ == "__main__":
    if mode == "train":
        data_analysis2.train("train-place.yaml")
    elif mode == "infer_visual":
        pass
    