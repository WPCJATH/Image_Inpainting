'''
The python script for data training of this specific inpainting task.

Author: WANG Xiangzhi
Date: 21-Dec-2023
'''

import data_analysis2
import data_visualization
data_path = "data/snowfield"

if __name__ == "__main__":
    data_analysis2.train("train-place.yaml")

    # Data Visualization after preprocessing and after inpainting
    data_source = "data/snowfield"
    data_dest = "data_visualization/"
    data_visualization.gen_display_preprocess_imgs(10, data_source, data_dest, display=False)
    data_visualization.gen_display_inpainted_imgs(10, data_dest, display=False)
    