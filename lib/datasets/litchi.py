# ------------------------------------------------------------------------------
# Litchi Dataset for DDRNet
# ------------------------------------------------------------------------------

import os
import cv2
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset

class Litchi(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_samples=None, 
                 num_classes=3,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=255, 
                 base_size=1024, 
                 crop_size=(512, 512), 
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]):

        super(Litchi, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std,)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        self.img_list = [line.strip().split() for line in open(list_path) if line.strip()]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

        # Class names for litchi dataset
        self.class_names = ['background', 'litchi', 'litchi_stem']

    def read_files(self):
        files = []
        for item in self.img_list:
            image_path = os.path.join(self.root, item[0])
            label_path = os.path.join(self.root, item[1])
            name = os.path.splitext(os.path.basename(item[0]))[0]
            files.append({
                "img": image_path,
                "label": label_path,
                "name": name,
            })
        return files

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        
        # Load image
        image = cv2.imread(item["img"], cv2.IMREAD_COLOR)
        
        # Load label
        label = np.array(Image.open(item["label"]))
        
        size = label.shape

        if self.multi_scale and np.random.uniform() > 0.5:
            # Multi-scale training
            scale = np.random.uniform(0.5, 2.0)
            h, w = int(size[0] * scale), int(size[1] * scale)
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)

        image = self.input_transform(image)
        image = image.transpose((2, 0, 1))

        if self.flip and np.random.uniform() > 0.5:
            image = image[:, :, ::-1]
            label = label[:, ::-1]

        # Random crop
        if image.shape[1] > self.crop_size[0] or image.shape[2] > self.crop_size[1]:
            h, w = image.shape[1], image.shape[2]
            pad_h = max(self.crop_size[0] - h, 0)
            pad_w = max(self.crop_size[1] - w, 0)
            if pad_h > 0 or pad_w > 0:
                image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), 'constant', constant_values=0)
                label = np.pad(label, ((0, pad_h), (0, pad_w)), 'constant', constant_values=self.ignore_label)
            
            h, w = image.shape[1], image.shape[2]
            start_h = np.random.randint(0, h - self.crop_size[0] + 1)
            start_w = np.random.randint(0, w - self.crop_size[1] + 1)
            image = image[:, start_h:start_h + self.crop_size[0], start_w:start_w + self.crop_size[1]]
            label = label[start_h:start_h + self.crop_size[0], start_w:start_w + self.crop_size[1]]
        else:
            # Pad to crop size
            h, w = image.shape[1], image.shape[2]
            pad_h = max(self.crop_size[0] - h, 0)
            pad_w = max(self.crop_size[1] - w, 0)
            if pad_h > 0 or pad_w > 0:
                image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), 'constant', constant_values=0)
                label = np.pad(label, ((0, pad_h), (0, pad_w)), 'constant', constant_values=self.ignore_label)

        return image.copy(), label.copy(), np.array(size), name

    def multi_scale_inference(self, config, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1,2,0)).copy()
        stride_h = int(self.crop_size[0] * 1.0)
        stride_w = int(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes,
                                    ori_height,ori_width]).cuda()
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                            rand_scale=scale,
                                            rand_crop=False)
            height, width = new_img.shape[:-1]
                
            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(config, model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = int(np.ceil(1.0 * new_h / stride_h))
                cols = int(np.ceil(1.0 * new_w / stride_w))
                preds = torch.zeros([1, self.num_classes,
                                           new_h,new_w]).cuda()
                count = torch.zeros([1,1, new_h, new_w]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(config, model, crop_img, flip)
                        preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                        count[:,:,h0:h1,w0:w1] += 1
                preds = preds / count
                preds = preds[:,:,:height,:width]

            preds = F.interpolate(
                preds, (ori_height, ori_width), 
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )            
            final_pred += preds
        return final_pred

    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = preds[i]
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))