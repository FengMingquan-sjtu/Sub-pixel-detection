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
    def __init__(self,input_HR_path,processed_HR_path,output_GT_path,output_LR_path,scale_factor,interestPointsDetect,max_corners):
        self.input_HR_path=input_HR_path
        self.processed_HR_path=processed_HR_path
        self.output_LR_path=output_LR_path
        self.output_GT_path=output_GT_path
        self.interestPointsDetect=interestPointsDetect
        self.scale_factor=scale_factor
        self.max_corners=max_corners

        if not os.path.exists(input_HR_path):
            raise IOError("input_HR_path= %s not exists"%input_HR_path)
        if not os.path.exists(processed_HR_path):
            os.makedirs(processed_HR_path)
        if not os.path.exists(output_GT_path):
            os.makedirs(output_GT_path)
        if not os.path.exists(output_LR_path):
            os.makedirs(output_LR_path)

    

    def dataPrepare(self):
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



        processed_HR_files=fileList(self.processed_HR_path)
        ## prepare GT
        for path,name in processed_HR_files:
            gray=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            GT_img=self.groundTruthGenerate(gray)
            cv2.imwrite(os.path.join(self.output_GT_path,name),GT_img)

        ## prepare LR

        for path,name in processed_HR_files:
            color=cv2.imread(path,cv2.IMREAD_COLOR)
            HR_height, HR_width, _ =color.shape
            LR_size=(int(HR_width/self.scale_factor),int(HR_height/self.scale_factor))
            LR_img=cv2.resize(color,LR_size)
            cv2.imwrite(os.path.join(self.output_LR_path,name),LR_img)

        print("Finish Preparation %d imgs to %s and %s"%(len(input_HR_files),self.output_GT_path,self.output_LR_path))

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
        """
        groundTruthGenerate: mark GT of HR_img,i.e. the interest points
        input:HR_img
        output:a 0-1 matrix of size HR_img, gray=255 points denoting detected points
        """
        interest_points=self.interestPointsDetect(HR_img,self.max_corners)
        #print("detected %d points"%(len(interest_points)))
        matrix=np.zeros(HR_img.shape)
        for i in interest_points:
            # notice the order of x,y is changed
            y=int(i[0,0])
            x=int(i[0,1])

            matrix[(x,y)]=GRAY_MAX

        return matrix



def cv2_interestDetect(gray,maxCorners):
    qualityLevel = 0.01
    minDistance = 10
    return cv2.goodFeaturesToTrack(gray, maxCorners, qualityLevel, minDistance)


if __name__ == '__main__':
    input_HR_path="../data/test_input"
    processed_HR_path='../data/processed_HR_temp'
    output_GT_path="../data/test_GT_output"
    output_LR_path="../data/test_LR_output"
    scale_factor=2
    interestPointsDetect=cv2_interestDetect
    max_corners=1000
    data=DataPrepare(input_HR_path,processed_HR_path,output_GT_path,output_LR_path,scale_factor,interestPointsDetect,max_corners)
    data.dataPrepare()



