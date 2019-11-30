import cv2
import numpy as np
import os
import sys
sys.path.append('../')
from utils.dataPrepare import fileList,GRAY_MAX


## base_output_path is the baseline model's output folder
class OutputVisualize:
    def __init__(self,test_GT_root,test_output_path,base_output_path, processed_output_path,threshold,detectors_order):
        self.test_GT_origin_path=os.path.join(test_GT_root,"img","origin")
        self.test_GT_paths=[os.path.join(test_GT_root,"img",d) for d in detectors_order]
        self.test_output_paths=[os.path.join(test_output_path,d) for d in detectors_order]
        self.base_output_paths=[os.path.join(base_output_path,d) for d in detectors_order]
        self.processed_output_paths=[os.path.join(processed_output_path,d) for d in detectors_order]
       
        for p in self.processed_output_paths:
            if not os.path.exists(p):
                os.makedirs(p)
        self.detectors_order=detectors_order
        self.threshold=threshold
        self.max_corners=1000

    #draw points on HR imgs (since we can not draw sub-pixel points on LR imgs)
    def visualize(self):
        print("start visualize")
        path_and_name=fileList(self.test_GT_origin_path)

        for path,name in path_and_name:
            input_path=path
            
            for i in range(len(self.detectors_order)):
                input_img=cv2.imread(input_path,cv2.IMREAD_COLOR)#color

                GT_path=os.path.join(self.test_GT_paths[i],name)
                test_path=os.path.join(self.test_output_paths[i],name)
                baseline_path=os.path.join(self.base_output_paths[i],name)

                GT_img=cv2.imread(GT_path,cv2.IMREAD_GRAYSCALE)#gray
                test_img=cv2.imread(test_path,cv2.IMREAD_GRAYSCALE)
                baseline_img=cv2.imread(baseline_path,cv2.IMREAD_GRAYSCALE)
                img_list=[GT_img,baseline_img,test_img]
                color_list=[[0,0,255],[225,0,0],[0,255,0]]#Red=GT,Blue=BASE,Green=TEST
            
                is_limit=self.detectors_order[i]=="corners"#corners are limited
                for idx_img in range(len(img_list)):
                    corner_list=self.getCornerList(img_list[idx_img],is_limit)
                    self.draw(input_img,corner_list,color_list[idx_img])
                cv2.imwrite(os.path.join(self.processed_output_paths[i],name),input_img)
        print("finish visualize")

    def draw(self,img,corner_list,color):
        size=1
        for x,y,_ in corner_list:#note that x,y is revert in cv2
            cv2.circle(img,(int(y),int(x)), size, color,-1)

    def getCornerList(self,img,is_limit=True):
        th=self.threshold*GRAY_MAX
        x,y=np.where(img>th)
        corner_list=[]
        for i in range(len(x)):
            corner_list.append((x[i],y[i],img[x[i],y[i]]))
        if is_limit and len(corner_list)>self.max_corners:
            #print(corner_list[:3])
            #print(corner_list[0][2])
            corner_list.sort(key=lambda t:t[2], reverse=True)
            return corner_list[:self.max_corners]
        else:
            return corner_list
        #print(idx.shape)




if __name__ == '__main__':
    ipt="../data/temp/test_processed_input"
    gt="../data/temp/test_GT"
    test="../data/output/test_output"
    base=test
    visualize="../data/output/visualize_output"
    threshold=0.9
    max_corners=1000
    detectors_order=["corners","edges"]
    o=OutputVisualize(ipt,gt,test,base,visualize,threshold,detectors_order)
    o.visualize()
