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
def gen_display_preprocess_imgs(n, data_source_path, data_out_path, random_pick = True):
    dataset = data_preprocessing.DeepImageDataset(data_source_path, return_raw=True)
    img_mask_path = os.path.join(data_out_path, "img_mask")
    if random_pick:
        indices = random.sample(range(len(dataset)), n)
    else:
        indices = range(n)

    images = []
    raw_imgs = []
    for i in indices:
        # Obtain the raw and inpainted images
        raw, image = dataset[i]
        he,wi = image.shape[:2]
        # Generate random masks
        masks = [
            data_preprocessing.generate_random_mask(he, wi, 0, 0, 128, 128)
            for _ in range(n)
        ]
        images.append(image)
        raw_imgs.append(raw)

    # Store the images and masks with names
    for i in range(n):
        data_preprocessing.save_image(images[i], os.path.join(img_mask_path, "img_" + str(i) + ".png"))
        data_preprocessing.save_image(masks[i], os.path.join(img_mask_path, "mask_" + str(i) + ".png"), is_gray=True)

    # Display the images and masks
    fig, axs = plt.subplots(n, 3)
    fig.suptitle("Image and Mask after the Pre-processing")
    for i in range(n):
        axs[i,0].imshow(raw_imgs[i])
        axs[i,1].imshow(images[i])
        axs[i,2].imshow(masks[i], cmap='gray')
        if i == 0:
            axs[i,0].set_title("Orginal")
            axs[i,1].set_title("Pre-Processed")
            axs[i,2].set_title("Mask")
    plt.show()



# Data Visualization after inpainting
def gen_display_inpainted_imgs(n, data_path):
    img_mask_path = os.path.join(data_path, "img_mask")
    non_deep_path = os.path.join(data_path, "non_deep_inpainted")
    deep_path = os.path.join(data_path, "deep_inpainted")
    os.makedirs(non_deep_path, exist_ok=True)
    os.makedirs(deep_path, exist_ok=True)

    images = []
    masks = []
    inpainted1 = []
    inpainted2 = []
    for i in range(n):
        # Load image and masks
        img_path = os.path.join(img_mask_path, "img_" + str(i) + ".png")
        mask_path = os.path.join(img_mask_path, "mask_" + str(i) + ".png")
        img = data_preprocessing.load_image_from_path(img_path)
        mask = data_preprocessing.load_image_from_path(mask_path, mode="gray")
        images.append(img)
        masks.append(mask)
        # Image inpainting
        inpainted1.append(data_analysis1.infer_image_non_deep_method(img, mask))
        data_preprocessing.save_image(inpainted1[i], os.path.join(non_deep_path, "img_" + str(i) + ".png"))
        inpainted2.append(
            data_analysis2.test(None, img_path, mask_path, os.path.join(deep_path, "img_" + str(i) + ".png"))
        )

    # Display with image, mask, inpainted1, inpainted2
    fig, axs = plt.subplots(n, 4)
    fig.suptitle("Inpainted Images")
    print("\n\n   Matrixes:")
    for i in range(n):
        # Print the matrixes
        print("Image", i, ":")
        print("Non deep Inpainted: l1 {} MSE {} PSNR {} SSIM {}".format(*data_analysis1.evaluation(inpainted1[i], images[i])))
        print("Deep Inpainted: l1 {} MSE {} PSNR {} SSIM {}".format(*data_analysis1.evaluation(inpainted2[i], images[i])))
        print()
        axs[i,0].imshow(images[i])
        axs[i,1].imshow(masks[i], cmap='gray')
        axs[i,2].imshow(inpainted1[i])
        axs[i,3].imshow(inpainted2[i])
        if i==0:
            axs[i,0].set_title("Pre-Processed")
            axs[i,1].set_title("Mask")
            axs[i,2].set_title("Non-deep Inpainted")
            axs[i,3].set_title("Deep Inpainted")
    plt.show()



if __name__ == "__main__":
    # Test
    data_source = "data/snowfield"
    data_dest = "data_visualization/"
    gen_display_preprocess_imgs(4, data_source, data_dest)
    gen_display_inpainted_imgs(4, data_dest)



