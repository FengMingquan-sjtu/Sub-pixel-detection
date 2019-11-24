import cv2
import os
import numpy as np
from torchvision import transforms as T
from PIL import Image

GRAY_MAX=255
HR_H=640
HR_W=640
def fileList(path):
    ret_list=[]
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(".png"): #modify to support other img types
                ret_list.append((os.path.join(root, name),name))
    return ret_list

class DataPrepare:
    def __init__(self,input_HR_path,processed_HR_path,output_GT_img_path,output_GT_npy_path,output_LR_path,scale_factor,detectors_order,detectors):
        self.input_HR_path=input_HR_path
        self.processed_HR_path=processed_HR_path
        self.output_LR_path=output_LR_path
        self.output_GT_npy_path=output_GT_npy_path
        self.detectors=detectors
        self.detectors_order=detectors_order
        self.scale_factor=scale_factor

        if not os.path.exists(input_HR_path):
            raise IOError("input_HR_path= %s not exists"%input_HR_path)
        if not os.path.exists(processed_HR_path):
            os.makedirs(processed_HR_path)

        if output_GT_img_path:
            self.output_GT_img_paths=[os.path.join(output_GT_img_path,d) for d in detectors_order]
            for p in self.output_GT_img_paths:
                if not os.path.exists(p):
                    os.makedirs(p)
        else:
            self.output_GT_img_paths=[] # no need for img output

        if not os.path.exists(output_GT_npy_path):
            os.makedirs(output_GT_npy_path)


        if not os.path.exists(output_LR_path):
            os.makedirs(output_LR_path)

    



    def dataPrepare(self):
        self.pre_process_spliting()
        #print("note: data is not split!")

        processed_HR_files=fileList(self.processed_HR_path)

        ## prepare GT
        for path,name in processed_HR_files:
            gray=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            #print(gray,path)
            GT=self.groundTruthGenerate(gray)
            for idx,p in enumerate(self.output_GT_img_paths):
                file=os.path.join(p,name)
                cv2.imwrite(file,GT[idx])
            p=self.output_GT_npy_path
            p=os.path.join(p,name+".npy")
            GT=np.array(GT)
            #print(GT.shape)
            np.save(p,GT)

        ## prepare LR

        for path,name in processed_HR_files:
            color=cv2.imread(path,cv2.IMREAD_COLOR)
            HR_height, HR_width, _ =color.shape
            LR_size=(int(HR_width/self.scale_factor),int(HR_height/self.scale_factor))
            LR_img=cv2.resize(color,LR_size)
            cv2.imwrite(os.path.join(self.output_LR_path,name),LR_img)

        print("Finish Preparation %d imgs"%(len(processed_HR_files)))

    def pre_process_spliting(self):
        ##extract files
        input_HR_files=fileList(self.input_HR_path)

        ##pre-process(split into fixed size, random transform)
        for path,name in input_HR_files:
            splited=self.split(path)
            for cnt,img in enumerate(splited):
                splited_name=str(cnt)+"_"+name
                img=self.transform(img)
                p=os.path.join(self.processed_HR_path,splited_name)
                img.save(p)
        print("pre-process spliting finished")

    def split(self,path):
        img=Image.open(path)
        w,h=img.size
        ret_list=list()
        for upper in range(0, h-HR_H ,HR_H ):
            lower=upper+HR_H
            for left in range(0, w-HR_W, HR_W):
                right=left+HR_W
                box=(left, upper, right, lower)
                croped=img.crop(box)
                ret_list.append(croped)
        return ret_list


    def transform(self,img):
        transforms = T.Compose([
            T.RandomHorizontalFlip(), #horizontal flip  with p=0.5
            T.RandomVerticalFlip(), #Vertical flip with p=0.5
            #T.RandomRotation(10), #random rotate in (-10,10),may lead to furry edge
            ])
        return transforms(img)

    def groundTruthGenerate(self,HR_img):
        return [self.detectors[d](HR_img) for d in self.detectors_order]




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
    matrix_edges = cv2.Canny(gray,150,220)
    return matrix_edges

def get_detectors(detector_num):
    detectors={"corners":cv2_cornersDetect,"edges":cv2_edgeDetect,"circles":cv2_circlesDetect}
    detectors_order=["corners","edges","circles"]
    return detectors_order[:detector_num],detectors


if __name__ == '__main__':
    input_HR_path="../data/input/test_input"
    processed_HR_path='../data/temp/test_processed_input'
    output_GT_img_path="../data/temp/test_GT/img"
    output_GT_npy_path="../data/temp/test_GT/npy"
    output_LR_path="../data/temp/test_LR"
    scale_factor=2
    detector_num=2
    detectors_order,detectors=get_detectors(detector_num)
    
    
    data=DataPrepare(input_HR_path,processed_HR_path,output_GT_img_path,output_GT_npy_path,output_LR_path,scale_factor,detectors_order,detectors)
    data.dataPrepare()



