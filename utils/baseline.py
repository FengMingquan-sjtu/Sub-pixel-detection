import cv2
import numpy as np
import os

from dataPrepare import fileList,cv2_interestDetect,GRAY_MAX

class Baseline:
    def __init__(self,test_input_path,baseline_output_path,max_corners):
        self.test_input_path=test_input_path
        self.baseline_output_path=baseline_output_path
        self.max_corners=max_corners
        if not os.path.exists(baseline_output_path):
            os.makedirs(baseline_output_path)
    def detect(self):
        pass