from PIL import Image
from torch.utils import data
from torchvision import transforms as T
import torch
import os

class DriverDataset(data.Dataset):
    def __init__(self, LR_root,GT_root,HR_height,HR_width,scale_factor):
        
        self.LR_root=LR_root
        self.GT_root=GT_root
        imgs_name = self._fileList(LR_root)
        self.imgs_name=imgs_name
        self.LR_transforms=self.default_transform(HR_height//scale_factor,HR_width//scale_factor)
        self.HR_transforms=self.default_transform(HR_height,HR_width)
        
    
    def __getitem__(self, index):
        name=  self.imgs_name[index]
        lr_path=os.path.join(self.LR_root, name)
        lr = self.LR_transforms(Image.open(lr_path))
        if self.GT_root==None:
            gt=name
        else:
            gt_path=os.path.join(self.GT_root, name)
            gt = self.HR_transforms(Image.open(gt_path))
        
        return (lr,gt)

    def __len__(self):
        return len(self.imgs_name)
    
    def _fileList(self,path):
        ret_list=[]
        for root, dirs, files in os.walk(path):
            for name in files:
                if name.endswith("png"):
                    ret_list.append(name)
        return ret_list

    def default_transform(self,H,W):
        transforms = T.Compose([
                T.CenterCrop(size=(H,W)),
                T.ToTensor(), 
                ])
        return transforms

if __name__ == '__main__':
    d=DriverDataset( LR_root="test_LR_output",GT_root="test_GT_output", transforms=None, isTrain=True)
    n=d.__getitem__(0)
    print(n)
    print(torch.sum(n[1]))