import cv2
import numpy as np
import os
import sys
sys.path.append('../')
from utils.dataPrepare import get_detectors,fileList


class Baseline:
    def __init__(self,test_LR_path,baseline_output_path,scale_factor,detectors_order):
        self.test_LR_path=test_LR_path
        self.baseline_output_path=baseline_output_path
        self.detectors_order=detectors_order
        self.scale_factor=scale_factor
        _,self.detectors=get_detectors(len(detectors_order))
        self.baseline_output_paths=[os.path.join(baseline_output_path,d) for d in self.detectors_order]
        for p in self.baseline_output_paths:
            if not os.path.exists(p):
                os.makedirs(p)
    def detect(self):
        for LR_path,name in fileList(self.test_LR_path):
            LR_img=cv2.imread(LR_path,cv2.IMREAD_GRAYSCALE)
            x,y=LR_img.shape
            HR_img=cv2.resize(LR_img,(x*self.scale_factor,y*self.scale_factor))
            for det_idx,det_name in enumerate(self.detectors_order):
                p=os.path.join(self.baseline_output_paths[det_idx],name)
                result=self.detectors[det_name](HR_img)
                cv2.imwrite(p,result)




if __name__ == '__main__':
    test_LR_path="../data/temp/test_LR"
    baseline_output_path="../data/output/baseline"
    scale_factor=2
    detectors_order=["corners","edges"]

    b=Baseline(test_LR_path,baseline_output_path,scale_factor,detectors_order)
    b.detect()