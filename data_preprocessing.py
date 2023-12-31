'''
The python script for data preprocessing of this specific inpainting task.

Author: WANG Xiangzhi
Date: 21-Dec-2023
'''
from PIL import Image
import numpy as np
import os, cv2
import torchvision.transforms as T
from torch.utils.data import Dataset

_valid_extensions = [".jpg", ".png", ".jpeg"]


def load_image_from_path(image_path, method="opencv", mode="RGB"):
    '''Load the image from file system'''
    if not os.path.exists(image_path):
        return None
    if not image_path.lower().endswith(tuple(_valid_extensions)):
        return None
    
    # For non-deep method
    if method == "opencv":
        image = cv2.imread(image_path)
        if mode == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if mode == "gray":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    # For deep method
    if method == "pil":
        image = Image.open(image_path)
        return image.convert(mode)

    return None

def save_image(image, image_path, is_gray=False):
    '''Save the image to file system'''
    if image.max() <= 1.0:
        image = (image * 255).astype('uint8')
    else:
        image = image.astype('uint8')
    if is_gray:
        image = Image.fromarray(image, "L")
    else:
        image = Image.fromarray(image)
    image.save(image_path)


def generate_random_mask(img_he, img_wi, margin_he, margin_wi, mask_hi, mask_wi, path=None):
    '''
    Generate a random mask 
    Reference: https://github.com/nipponjo/deepfillv2-pytorch/blob/master/utils/misc.py
    '''
    maxt = img_he - margin_he - mask_hi
    maxl = img_wi - margin_wi - mask_wi
    t = np.random.randint(margin_he, maxt)
    l = np.random.randint(margin_wi, maxl)
    mask = np.zeros((img_he, img_wi), np.uint8)
    mask[t:t+mask_hi, l:l+mask_wi] = 255
    if path != None:
        cv2.imwrite(path, mask)
    return mask

def perform_random_crop(image):
    '''
    Perform random crop on the image
    Reference: https://github.com/nipponjo/deepfillv2-pytorch/blob/master/train.py
    '''
    return T.RandomCrop(image.shape[1:])(image)


class DeepImageDataset(Dataset):
    '''
    The dataset class for loading images on the fly.
    Reference: https://github.com/nipponjo/deepfillv2-pytorch/blob/master/utils/data.py
    '''
    def __init__(self, folder_path, use_tensor=False, return_raw=False):
        super().__init__()
        self.images = [os.path.join(folder_path, image_name) for image_name in os.listdir(folder_path)]
        self.use_tensor = use_tensor
        self.return_raw = return_raw

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = load_image_from_path(self.images[index], method="pil")
        prosed_image = T.ToTensor()(image.copy())
        prosed_image = perform_random_crop(prosed_image)
        if not self.use_tensor:
            prosed_image = prosed_image.permute(1, 2, 0).contiguous().cpu().numpy()
        if self.return_raw:   
            return image, prosed_image
        return prosed_image

if __name__ == "__main__":
    # Test
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    dataset = DeepImageDataset("data/", True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in dataloader:
        print(batch.shape)
        for image in batch:
            plt.imshow(image.permute(1, 2, 0))
            plt.show()
        break