import cv2
import numpy as np
import os

from dataPrepare import fileList,GRAY_MAX



class OutputProcessor:
    def __init__(self,test_input_path,test_GT_path,test_output_path,baseline_output_path, processed_output_path,threshold):
        self.test_input_path=test_input_path
        self.test_GT_path=test_GT_path
        self.test_output_path=test_output_path
        self.baseline_output_path=baseline_output_path
        self.processed_output_path=processed_output_path
        if not os.path.exists(processed_output_path):
            os.makedirs(processed_output_path)
        #self.max_corners=max_corners
        self.threshold=threshold

    def process(self):
        path_and_name=fileList(self.test_input_path)
        for path,name in path_and_name:
            input_path=path
            GT_path=os.path.join(self.test_GT_path,name)
            test_path=os.path.join(self.test_output_path,name)
            baseline_path=os.path.join(self.baseline_output_path,name)

            input_img=cv2.imread(input_path,1)#color
            #print(input_img.shape)
            GT_img=cv2.imread(GT_path,0)#gray
            #print(GT_img.shape)
            test_img=cv2.imread(test_path,0)
            baseline_img=cv2.imread(baseline_path,0)
            img_list=[GT_img,test_img,baseline_img]
            color_list=[[0,0,255],[0,255,0],[225,0,0]]#RGB

            for i in range(len(img_list)):
                corner_list=self.getCornerList(img_list[i])
                self.draw(input_img,corner_list,color_list[i])
            cv2.imwrite(os.path.join(self.processed_output_path,name),input_img)

    def draw(self,img,corner_list,color):
        size=4
        for x,y in corner_list:#note that x,y is revert in cv2
            cv2.circle(img,(int(y),int(x)), size, color,-1)

    def getCornerList(self,img):
        th=self.threshold*GRAY_MAX
        x,y=np.where(img>th)
        corner_list=[]
        for i in range(len(x)):
            corner_list.append((x[i],y[i]))
        return corner_list
        #print(idx.shape)




if __name__ == '__main__':
    ipt="../data/test_input"
    gt="../data/test_GT"
    test="../data/test_output"
    base=test
    processed="../data/processed_output"
    max_corners=1000
    threshold=0.5
    o=OutputProcessor(ipt,gt,test,base,processed,max_corners,threshold)
    o.process()
