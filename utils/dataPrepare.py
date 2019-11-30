import cv2
import os
import numpy as np
from torchvision import transforms as T
from PIL import Image

GRAY_MAX=255
HR_H=640
HR_W=640
ratio=1.23
crop_H=int(ratio*HR_H)
crop_W=int(ratio*HR_W)
stride=0.8
stride_H=int(stride*HR_H)
stride_W=int(stride*HR_W)

def fileList(path):
    ret_list=[]
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(".png"): #modify to support other img types
                ret_list.append((os.path.join(root, name),name))
    return ret_list

class DataPrepare:
    def __init__(self,input_HR_path,output_GT_root,output_LR_root,scale_factor,detectors_order,detectors,need_visualize=False):
        self.input_HR_path=input_HR_path
        self.output_LR_root=output_LR_root
        self.output_GT_root=output_GT_root
        self.detectors=detectors
        self.detectors_order=detectors_order
        self.scale_factor=scale_factor
        self.need_visualize=need_visualize

        self.transform=self.default_transform()

        self.LR_size=(int(HR_W/self.scale_factor),int(HR_H/self.scale_factor))

        if not os.path.exists(input_HR_path):
            raise IOError("input_HR_path= %s not exists"%input_HR_path)



        self.output_GT_img_paths=[os.path.join(output_GT_root,"img",d) for d in detectors_order]
        self.output_GT_img_paths.append(os.path.join(output_GT_root,"img","origin"))
        for p in self.output_GT_img_paths:
            if not os.path.exists(p):
                os.makedirs(p)
        self.output_LR_img_paths=[os.path.join(output_LR_root,"img",d) for d in detectors_order]
        self.output_LR_img_paths.append(os.path.join(output_LR_root,"img","origin"))
        for p in self.output_LR_img_paths:
            if not os.path.exists(p):
                os.makedirs(p)


        
        self.output_GT_npy_path=os.path.join(output_GT_root,"npy")
        self.output_LR_npy_path=os.path.join(output_LR_root,"npy")
        
        if not os.path.exists(self.output_LR_npy_path):
            os.makedirs(self.output_LR_npy_path)
        if not os.path.exists(self.output_GT_npy_path):
            os.makedirs(self.output_GT_npy_path)

    



    def dataPrepare(self):
        self.pre_process_spliting()
        #print("note: data is not split!")
        HR_origin_path=self.output_GT_img_paths[len(self.detectors_order)]# the path stores original HR
        HR_origin_files=fileList(HR_origin_path)
        ## prepare GT
        self.process_and_save(HR_origin_files,self.output_GT_img_paths,self.output_GT_npy_path)


        ## prepare LR
        LR_origin_path=self.output_LR_img_paths[len(self.detectors_order)]
        for path,name in HR_origin_files:
            color=cv2.imread(path,cv2.IMREAD_COLOR)
            LR_img=cv2.resize(color,self.LR_size)
            cv2.imwrite(os.path.join(LR_origin_path,name),LR_img)
            #if self.need_visualize:
        LR_origin_files=fileList(LR_origin_path)
        self.process_and_save(LR_origin_files,self.output_LR_img_paths,self.output_LR_npy_path)
                

        print("Finish Preparation %d imgs"%(len(HR_origin_files)))

    def process_and_save(self,files,img_paths,npy_path):
        
        for path,name in files:
            gray=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            #print(gray,path)
            GT=self.detect(gray)
            if self.need_visualize:
                for idx in range(len(self.detectors_order)):
                    file=os.path.join(img_paths[idx],name)
                    cv2.imwrite(file,GT[idx])

            p=os.path.join(npy_path,name+".npy")
            GT=np.array(GT)
            #print(GT.shape)-->(C,H,W) here C==|detectors|==2
            color=cv2.imread(path,cv2.IMREAD_COLOR)
            #print(color.shape)--> (H,W,C);  Here C==|colors|==3
            color=color.transpose((2,0,1))
            #print(color.shape)#--> (C,H,W)
            GT=np.concatenate((GT,color),axis=0)
            #print(GT.shape) --> (C,H,W),C=5
            np.save(p,GT)

    def pre_process_spliting(self):
        ##extract files
        input_HR_files=fileList(self.input_HR_path)

        ##pre-process(split into fixed size, random transform)
        HR_origin_path=self.output_GT_img_paths[len(self.detectors_order)]# the path stores original HR
        for path,name in input_HR_files:
            splited=self.split(path)
            for cnt,img in enumerate(splited):
                splited_name=str(cnt)+"_"+name
                img=self.transform(img)
                p=os.path.join(HR_origin_path,splited_name)
                img.save(p)
        print("pre-process spliting finished")

    def split(self,path):
        img=Image.open(path)
        w,h=img.size
        ret_list=list()
        
        for upper in range(0, h-crop_H ,stride_H):
            lower=upper+crop_H
            for left in range(0, w-crop_W, stride_W):
                right=left+crop_W
                box=(left, upper, right, lower)
                croped=img.crop(box)
                ret_list.append(croped)
        return ret_list


    def default_transform(self):
        transforms = T.Compose([
            T.RandomRotation(10), #random rotate in (-10,10),may lead to furry edge
            T.CenterCrop((HR_H,HR_W)),
            T.RandomHorizontalFlip(), #horizontal flip  with p=0.5
            T.RandomVerticalFlip(), #Vertical flip with p=0.5
            ])
        return transforms

    def detect(self,img):
        return [self.detectors[d](img) for d in self.detectors_order]




def cv2_cornersDetect(gray):
    max_corners=1000
    qualityLevel = 0.01
    minDistance = 10
    corners=cv2.goodFeaturesToTrack(gray, max_corners, qualityLevel, minDistance)
    matrix_corners=np.zeros(gray.shape)
    for i in corners:
        # notice the order of x,y is changed
        y=int(i[0,0])
        x=int(i[0,1])
        matrix_corners[x,y]=GRAY_MAX
    return matrix_corners


def cv2_circlesDetect(gray):
    matrix_circles=np.zeros(gray.shape)
    circles=cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20)
    if isinstance(circles, np.ndarray):
        circles=circles[0]
        for i in circles:
            cv2.circle(matrix_circles,(i[0],i[1]),i[2],255,1)
    return matrix_circles

def cv2_edgeDetect(gray):
    matrix_edges = cv2.Canny(gray,100,200)
    return matrix_edges

def get_detectors(detector_num):
    detectors={"corners":cv2_cornersDetect,"edges":cv2_edgeDetect,"circles":cv2_circlesDetect}
    detectors_order=["corners","edges","circles"]
    return detectors_order[:detector_num],detectors


if __name__ == '__main__':
    input_HR_path="../data/input/test_input"
    output_GT_root="../data/temp/test_GT"
    output_LR_root="../data/temp/test_LR"
    scale_factor=2
    detector_num=2
    detectors_order,detectors=get_detectors(detector_num)
    
    need_visualize=True
    data=DataPrepare(input_HR_path,output_GT_root,output_LR_root,scale_factor,detectors_order,detectors,need_visualize)
    data.dataPrepare()



