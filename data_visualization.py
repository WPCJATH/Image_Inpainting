'''
The python script for data visualization of this specific inpainting task.

Author: WANG Xiangzhi
Date: 21-Dec-2023
'''
import random, os
import data_preprocessing, data_analysis1, data_analysis2
import matplotlib.pyplot as plt


os.makedirs("data_visualization", exist_ok=True)

# Data Visualization after preprocessing
def gen_display_preprocess_imgs(n, data_path, random_pick = True):
    dataset = data_preprocessing.DeepImageDataset(data_path)
    if random_pick:
        indices = random.sample(range(len(dataset)), n)
    else:
        indices = range(n)

    images = []
    for i in indices:
        image = dataset[i]
        he,wi = image.shape[:2]
        masks = [
            data_preprocessing.generate_random_mask(he, wi, 0, 0, 128, 128)
            for _ in range(n)
        ]
        images.append(image)

    # Store the images and masks with names
    for i in range(n):
        data_preprocessing.save_image(images[i], "data_visualization/img_" + str(i) + ".png")
        data_preprocessing.save_image(masks[i], "data_visualization/mask_" + str(i) + ".png", True)

    # Display the images and masks
    _, axs = plt.subplots(n, 2)
    for i in range(n):
        axs[i,0].imshow(images[i])
        axs[i,1].imshow(masks[i], cmap='gray')
    plt.show()



# Data Visualization after inpainting
def gen_display_inpainted_imgs(n, data_path):
    non_deep_path = os.path.join(data_path, "non_deep_inpainted")
    deep_path = os.path.join(data_path, "deep_inpainted")
    os.makedirs(non_deep_path, exist_ok=True)
    os.makedirs(deep_path, exist_ok=True)

    images = []
    masks = []
    inpainted1 = []
    inpainted2 = []
    for i in range(n):
        img_path = os.path.join(data_path, "img_" + str(i) + ".png")
        mask_path = os.path.join(data_path, "mask_" + str(i) + ".png")
        img = data_preprocessing.load_image_from_path(img_path)
        mask = data_preprocessing.load_image_from_path(mask_path, mode="gray")
        images.append(img)
        masks.append(mask)
        inpainted1.append(data_analysis1.infer_image_non_deep_method(img, mask))
        data_preprocessing.save_image(inpainted1[i], os.path.join(non_deep_path, "img_" + str(i) + ".png"))
        inpainted2.append(
            data_analysis2.test(None, img_path, mask_path, os.path.join(deep_path, "img_" + str(i) + ".png"))
        )

    # Display with image, mask, inpainted1, inpainted2
    _, axs = plt.subplots(n, 4)
    for i in range(n):
        axs[i,0].imshow(images[i])
        axs[i,1].imshow(masks[i], cmap='gray')
        axs[i,2].imshow(inpainted1[i])
        axs[i,3].imshow(inpainted2[i])
    plt.show()



if __name__ == "__main__":
    # Test
    gen_display_preprocess_imgs(4, "data/snowfield")
    gen_display_inpainted_imgs(4, "data_visualization/")



