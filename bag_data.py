from ensurepip import bootstrap
from glob import iglob

import torch
from pandas import read_excel
from PIL import Image
from torchvision import transforms

transform = transforms.Compose(
    [
        transforms.Resize(256, 256),
        transforms.ToTensor(),
        # Potential Normalize
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
        self.load_patch(patches_folder)
        if is_bootstrap:
            self.bootstrap()
        else:
            pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

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

    def bootstrap(self):
        tmp_patch_list = self.pos_patch_list + self.neg_patch_list
        for patch in tmp_patch_list:
            pass
