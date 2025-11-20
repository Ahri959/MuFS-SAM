# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from .common import LayerNorm2d
from typing import Any, Dict, List, Tuple
from src.segment_anything.utils.transforms import ResizeLongestSide
from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from.MFM import MultiModalFeaturePool,MMA
from .utils import prepare_image, extract_bboxes_expand, extract_points, extract_mask


class VariableKernelDownsampling(nn.Module):
    def __init__(self, in_channels=768, out_channels=3):
        super(VariableKernelDownsampling, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(512, 128, kernel_size=1, stride=2, padding=0)
        self.conv3 = nn.Conv2d(128, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x



class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.img_down = VariableKernelDownsampling()

        self.MMA = MMA()
        self.mask_down = nn.Sequential(
            nn.Conv2d(1, 4,kernel_size=3, stride=2, padding=1),
            nn.Conv2d(4, 1,kernel_size=3, stride=2, padding=1)
        )

        self.fusion_downscaling = nn.Sequential(
            nn.Conv2d(256*2+1, 256, kernel_size=3, stride=1, padding=1),  # 融合特征
            LayerNorm2d(256),
            nn.GELU(),
            
            nn.Conv2d(256, 192, kernel_size=3, stride=1, padding=1),  # 通道数减少
            LayerNorm2d(192),
            nn.GELU(),
            
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),  # 通道数再次减少
            LayerNorm2d(128),
            nn.GELU(),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 恢复到 256 通道
        )

        # 上采样层，将特征图恢复到输入的大小
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)



    def sam_input_prepare(self,image, pred_masks, image_embeddings=None, resize_transform=None, use_point=True, use_box=True, use_mask=True, add_neg=True, margin=0.0, gamma=1.0, strength=15):
        ori_size = pred_masks.shape[-2:]
        input_dict = {
            'image': image,
            'original_size': ori_size,
            }
        
        target_size = image.shape[1:]
        expand_list = torch.zeros((len(pred_masks))).to(image.device)
        if use_box:
            bboxes, box_masks, areas, expand_list = extract_bboxes_expand(image_embeddings, pred_masks, margin=margin)
            input_dict['boxes'] = resize_transform.apply_boxes_torch(bboxes, ori_size)
        
        point_coords, point_labels, gaus_dt = extract_points(pred_masks, add_neg=add_neg, use_mask=use_mask, gamma=gamma)
        if use_point:
            input_dict['point_coords'] = resize_transform.apply_coords_torch(point_coords, ori_size)
            input_dict['point_labels'] = point_labels
            
      
        return bboxes[0],[point_coords,point_labels]


    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"].squeeze(0)) for x in batched_input], dim=0)
        input_thermal = torch.stack([self.preprocess(x["depth"].squeeze(0)) for x in batched_input], dim=0)
        image_embeddings,fusion_embeddings = self.image_encoder(input_images,input_thermal)

        outputs = []
        for image_record, curr_embedding,fusion_embedding in zip(batched_input, image_embeddings,fusion_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                curr_embedding = None,
                points=points,
                boxes=None,
                masks=image_record.get("tensor_depth", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )


            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            for i in range(2):
                low_res_mask_reshaped = masks
                masks = masks > self.mask_threshold

                
                masks = masks.float()
                fore_mask = masks
                back_mask = 1-masks
                fore_mask = self.mask_down(masks)
                back_mask = self.mask_down(back_mask)
                

                img = fusion_embedding.unsqueeze(0)

                fore_img = img*fore_mask
                back_img = img*back_mask

                fore_img = torch.cat([img,fore_img,fore_mask],dim = 1)
                back_img = torch.cat([img,back_img,back_mask],dim = 1)

                fore_img = self.fusion_downscaling(fore_img)
                back_img = self.fusion_downscaling(back_img)

                fore_img = fore_img + img
                back_img = back_img + img

                dense_fusion_embeddings = self.MMA(fore_img,back_img) + dense_embeddings

                low_res_masks, iou_predictions = self.mask_decoder(
                    image_embeddings=curr_embedding.unsqueeze(0),
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_fusion_embeddings,
                    multimask_output=multimask_output,
                )

                masks = self.postprocess_masks(
                    low_res_masks,
                    input_size=image_record["image"].shape[-2:],
                    original_size=image_record["original_size"],
                )

            low_res_mask_reshaped = masks
            masks = masks > self.mask_threshold
          
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_mask_reshaped,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """

        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]

        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
