from glob import iglob
from random import random
from unittest import result

import numpy as np
import torch
from pandas import read_excel
from PIL import Image
from torchvision import transforms

transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


class Bag_WSI(torch.utils.data.Dataset):
    def __init__(
        self,
        excel_path,
        patches_path,
        classification_label,
        positive_label,
        bag_size,
        is_bootstrap=True,
    ):
        self.classification_label = classification_label
        self.positive_label = positive_label
        self.bag_size = bag_size

        print("load csv")
        csv_data = read_excel(excel_path)
        patches_folder = sorted(
            iglob(patches_path + "/*"), key=lambda x: int(x.split("/")[-1])
        )

        self.patch_label = list(
            map(
                lambda x: 1 if x == self.positive_label else 0,
                csv_data[self.classification_label],
            )
        )
        self.pos_patch_list = []
        self.neg_patch_list = []
        self.pos_bag_list = []
        self.neg_bag_list = []
        print("load patch")
        self.load_patch(patches_folder)
        print("check bootstrap")
        if is_bootstrap:
            self.neg_bag_list = self.bootstrap(self.neg_patch_list)
            self.pos_bag_list = self.bootstrap(self.pos_patch_list)
        else:
            pass
        self.bag_list = []
        self.bag_list.extend(
            zip(self.pos_bag_list, torch.ones(len(self.pos_bag_list), 1))
        )
        self.bag_list.extend(
            zip(self.neg_bag_list, torch.zeros(len(self.neg_bag_list), 1))
        )

    def __len__(self):
        return len(self.bag_list)

    def __getitem__(self, index):
        return self.bag_list[index]

    def load_patch(self, patches_folder):
        for index, folder in enumerate(patches_folder):
            files = iglob(folder + "/*")
            image_list = []
            for file in files:
                image = transform(Image.open(file))
                image_list.append(image)
            patch_tensor = torch.stack(image_list, axis=0)
            if self.patch_label[index]:
                self.pos_patch_list.append(patch_tensor)
            else:
                self.neg_patch_list.append(patch_tensor)

    def bootstrap(self, bag_list):
        result_patch_list = []
        count = 0
        for patch in bag_list:
            count += 1
            if len(patch) < self.bag_size:
                random_num = torch.randperm(self.bag_size - len(patch))
                random_num = list(
                    map(
                        lambda x: x if x < len(patch) else x % len(patch),
                        random_num.tolist(),
                    )
                )
                # print('patch len is',len(patch),'add num is',len(random_num))
                stack_list = []
                for idx in random_num:
                    tmp_image = patch[idx]
                    stack_list.append(tmp_image)
                stacked = torch.stack(stack_list, dim=0)
                patch = torch.cat([patch, stacked], dim=0)
                result_patch_list.append(patch)
            else:
                bootstrap_times = self.bag_size
                for rep in range(bootstrap_times):
                    img_list = []
                    random_num = torch.randperm(len(patch)).tolist()[: self.bag_size]
                    for idx in random_num:
                        img_list.append(patch[idx])
                    patch_tensor = torch.stack(img_list, dim=0)
                    result_patch_list.append(patch_tensor)
        return result_patch_list
