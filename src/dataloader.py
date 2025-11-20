import torch
import glob
import os 
from PIL import Image
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b, SamPredictor
from src.lora import LoRA_sam
import src.utils as utils
import yaml
from PIL import Image
import random
import torchvision.transforms as transforms

class DatasetSegmentation(Dataset):
    """
    Dataset to process the images and masks

    Arguments:
        folder_path (str): The path of the folder containing the images
        processor (obj): Samprocessor class that helps pre processing the image, and prompt 
    
    Return:
        (dict): Dictionnary with 4 keys (image, original_size, boxes, ground_truth_mask)
            image: image pre processed to 1024x1024 size
            original_size: Original size of the image before pre processing
            boxes: bouding box after adapting the coordinates of the pre processed image
            ground_truth_mask: Ground truth mask
    """

    def __init__(self, config_file: dict, processor: Samprocessor, mode: str):
        super().__init__()
        if mode == "train":
            self.img_files = glob.glob(os.path.join(config_file["DATASET"]["TRAIN_PATH"],'images',"*"+config_file["DATASET"]["IMAGE_FORMAT"]))
            self.mask_files = []
            self.prompt_files = []
            self.depth_files = []
            for img_path in self.img_files:
                self.mask_files.append(os.path.join(config_file["DATASET"]["TRAIN_PATH"],'masks', os.path.basename(img_path)[:-4] + config_file["DATASET"]["IMAGE_FORMAT"]))
            for img_path in self.img_files:
                self.prompt_files.append(img_path.replace("images",config_file["DATASET"]["Modal"]))
            for img_path in self.img_files:
                self.depth_files.append(img_path.replace("images",config_file["DATASET"]["Modal"]))

        else:
            self.img_files = glob.glob(os.path.join(config_file["DATASET"]["TEST_PATH"],'images',"*"+config_file["DATASET"]["IMAGE_FORMAT"]))
            self.mask_files = []
            self.prompt_files = []
            self.depth_files = []
            for img_path in self.img_files:
                self.mask_files.append(os.path.join(config_file["DATASET"]["TEST_PATH"],'masks', os.path.basename(img_path)[:-4] + config_file["DATASET"]["IMAGE_FORMAT"]))
            for img_path in self.img_files:
                self.prompt_files.append(img_path.replace("images",config_file["DATASET"]["Modal"]))
            for img_path in self.img_files:
                self.depth_files.append(img_path.replace("images",config_file["DATASET"]["Modal"]))

        self.processor = processor


    def rotate_image(self,image, angle):
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))


    def augment(self, image, flipCode):
        """
        数据增强函数，根据 flipCode 执行不同增强方式：
        - 1: 水平翻转
        - 0: 垂直翻转
        - -1: 水平 + 垂直翻转
        - 2: 顺时针旋转 90 度
        - 3: 高斯模糊
        - 4: 调整亮度（增强）
        - 5: 添加固定高斯噪声
        - 6: 颜色抖动（ColorJitter）
        - 7: 随机裁剪（Random Crop）
        - 8: 仿射变换（Affine）
        - 9: 随机通道交换（Channel Shuffle）
        - 10: 添加椒盐噪声（Salt & Pepper Noise）
        - 11: 对比度拉伸（Contrast Stretching）

        参数：
        - image: 输入图像，PIL 格式
        - flipCode: 增强模式的选择代码

        返回：
        - 增强后的图像，PIL 格式
        """
        # 转换为 numpy 格式
        image = np.array(image)

        if flipCode == 1:
            # 水平翻转
            image = cv2.flip(image, 1)
        elif flipCode == 0:
            # 垂直翻转
            image = cv2.flip(image, 0)
        elif flipCode == -1:
            # 水平 + 垂直翻转
            image = cv2.flip(image, -1)
        elif flipCode == 2:
            # 顺时针旋转 90 度
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif flipCode == 3:
            # 高斯模糊
            image = self.rotate_image(image, 45) 
        elif flipCode == 4:
            # 调整亮度（增强）
            image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
        elif flipCode == 5:
            # 添加固定高斯噪声
     
            image = self.rotate_image(image, 135)
        elif flipCode == 6:
            # 颜色抖动
            image = self.rotate_image(image, 225)
        elif flipCode == 7:
            # 随机裁剪
            image = self.rotate_image(image, 315)
        elif flipCode == 8:
            # 仿射变换
            image = self.rotate_image(image, 30)
        elif flipCode == 9:
            image = self.rotate_image(image, 60)
        elif flipCode == 10:
            image = self.rotate_image(image, 120)
        elif flipCode == 11:
            image = self.rotate_image(image, 150)
        elif flipCode == 12:
            image = self.rotate_image(image, 210)
        elif flipCode == 13:
            image = self.rotate_image(image, 240)
        elif flipCode == 14:
            image = self.rotate_image(image, 300)
        elif flipCode == 15:
            image = self.rotate_image(image, 330)





        # 转回 PIL 格式
        return Image.fromarray(image)

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index: int) -> list:
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]#.replace("jpg","png")
            prompt_path = self.prompt_files[index]#.replace("jpg","png")
            depth_path = self.depth_files[index]#.replace("jpg","png")


            # get image and mask in PIL format
            image = Image.open(img_path).resize((256, 256))

            mask = Image.open(mask_path).resize((256, 256))
            prompt = Image.open(prompt_path).convert('L').resize((256, 256))
            depth = Image.open(depth_path).convert('RGB').resize((256, 256))
            tensor_img = image
            if "train" in img_path:
                flipCode = random.choice(list(range(-1, 7)))  # 随机选择 1~11 之间的增强方法
                if flipCode in [1, 0, -1, 2,3,5,6,7,8,9,10,11,12,13,14,15]:
                    image = self.augment(image, flipCode)
                    mask = self.augment(mask, flipCode)
                    prompt = self.augment(prompt, flipCode)
                    depth = self.augment(depth, flipCode)
                    tensor_img = self.augment(tensor_img, flipCode)
                elif flipCode in [4]:
                # , 6, 7, 8, 9, 10, 11]:
                    image = self.augment(image, flipCode)
            
            mask = mask.convert('1')    
            ground_truth_mask = np.array(mask)
            transform = transforms.ToTensor()

            original_size = tuple(image.size)[::-1]
    
            # get bounding box prompt

            inputs = self.processor(image, depth,original_size)
            inputs["ground_truth_mask"] = torch.from_numpy(ground_truth_mask).to('cuda')
            inputs["prompt"] = transform(prompt).to('cuda')
            inputs["tensor_depth"] = transform(depth).to('cuda')*255
            inputs["tensor_img"] = transform(tensor_img).to('cuda')


            return inputs
    
def collate_fn(batch: torch.utils.data) -> list:
    """
    Used to get a list of dict as output when using a dataloader

    Arguments:
        batch: The batched dataset
    
    Return:
        (list): list of batched dataset so a list(dict)
    """
    return list(batch)