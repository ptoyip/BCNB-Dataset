from PIL import Image
import torch
from torchvision import transforms
from glob import iglob
from pandas import read_excel

class Bag_WSI():
    # Bag => WSI image cropped shuffled set
    # Instance => mixed_patches
    def __init__(
        self,
        excel_path=r"patient-clinical-data.xlsx",
        patches_path=r"patches",
        classification_label="ER",
        positive_label="Positive",
        bag_size=30,  # Median of patches size is 36.5 FYI
    ) -> None:
        self.csv_data = read_excel(excel_path)
        self.patch_folder = sorted(
            iglob(patches_path + "/*"), key=lambda x: int(x.split("/")[-1])
        )
        self.patch_label = list(
            map(
                lambda x: 1 if x == positive_label else 0,
                self.csv_data[classification_label],
            )
        )
        self.bag_size = bag_size
        print("Initialising")
        self.__patch_labelling()
        print("finished generation")
        self.bag_list = list(
            [
                *zip(self.pos_bag_list, torch.ones(len(self.pos_bag_list),1)),
                *zip(self.neg_bag_list, torch.zeros(len(self.neg_bag_list),1)),
            ]
        )

    def __patch_labelling(self):
        unlabeled_img_list = []
        self.pos_bag_list = []
        self.neg_bag_list = []
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                # ? Need normalised the image?
            ]
        )
        for folder in self.patch_folder:
            patch_img = []
            file_path = iglob(folder + "/*")
            for img_path in file_path:
                patch_img.append(transform(Image.open(img_path)))
            unlabeled_img_list.append(torch.stack(patch_img, dim=0))
        for patch_idx, patch in enumerate(unlabeled_img_list): # patch is tensor
            # * Check Patch Tensor Length
            if len(patch) <= self.bag_size:
                # * Generate a bag directly, can write a function to make it look better
                if self.patch_label[patch_idx]:
                    self.pos_bag_list.append(patch)
                else:
                    self.neg_bag_list.append(patch)
            else:
                # * Generate multiple bags with random sampled instances, number of bags depend on the number of instances the patch have.
                bootstrap_times = len(patch) // self.bag_size + 1
                for i in range(bootstrap_times):
                    rand_int = torch.multinomial(
                        torch.ones(self.bag_size, dtype=torch.float),
                        self.bag_size,
                        replacement=True,
                    )
                    bag = patch[rand_int]  # 1 * N(img) * 3 * 255 * 255
                    if self.patch_label[patch_idx]:
                        self.pos_bag_list.append(bag)
                    else:
                        self.neg_bag_list.append(bag)
        
