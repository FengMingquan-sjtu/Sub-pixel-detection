#python modules
import os
from PIL import Image

#torch modules
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torch.nn.functional as F
import torch.optim as optim
import torch

#my modules
from utils.trainer import Trainer,Tester
from utils.driverDatasetSR import DriverDatasetSR
from utils.dataPrepare import DataPrepare,HR_H,HR_W,get_detectors
from utils.outputVisualize import OutputVisualize
from utils.option import getArgs
from utils.baseline import Baseline
from models.model import SPResNet

# -----begin set parameters----------
args=getArgs()
batch_size=args.batch_size
max_epoch=args.max_epoch
step_size=args.step_size #learning rate update step
scale_factor=args.scale_factor #scale_factor must be a factor of HR_H and HR_W
detector_num=args.detector_num
feature_size=args.feature_size
num_ResBlock=args.num_ResBlock
max_corners=args.max_corners
threshold=args.threshold
model_name=args.model_name
experiment_name=args.experiment_name
load_epoch_pkl=args.load_epoch_pkl
isTrain=args.isTrain
isTest=args.isTest
isVisualize=args.isVisualize
isPrepare=args.isPrepare
isSummary=args.isSummary


detectors_order,detectors=get_detectors(detector_num)


data_root="data"
input_root="input"
temp_root="temp"
output_root="output"

valid_input_HR_path=os.path.join(data_root,input_root,"valid_input")
valid_processed_HR_path=os.path.join(data_root,temp_root,"valid_processed_input")
#valid_GT_npy_path=os.path.join(data_root,temp_root,"valid_GT","npy")
#valid_GT_img_path=os.path.join(data_root,temp_root,"valid_GT","img")
valid_LR_path=os.path.join(data_root,temp_root,"valid_LR")

train_input_HR_path=os.path.join(data_root,input_root,"train_input")
train_processed_HR_path=os.path.join(data_root,temp_root,"train_processed_input")
#train_GT_npy_path=os.path.join(data_root,temp_root,"train_GT","npy")
#train_GT_img_path=os.path.join(data_root,temp_root,"train_GT","img")
train_LR_path=os.path.join(data_root,temp_root,"train_LR")

test_input_HR_path=os.path.join(data_root,input_root,"test_input")
test_processed_HR_path=os.path.join(data_root,temp_root,"test_processed_input")
#test_GT_npy_path=os.path.join(data_root,temp_root,"test_GT","npy")
#test_GT_img_path=os.path.join(data_root,temp_root,"test_GT","img")
test_LR_path=os.path.join(data_root,temp_root,"test_LR")
test_output_path=os.path.join(data_root,output_root,"test_output")

baseline_output_path=os.path.join(data_root,output_root,"baseline_output")

visualize_output_path=os.path.join(data_root,output_root,"visualize_output")

log_root="logs"
log_path=os.path.join("logs",experiment_name)
model_save_path=os.path.join("weight",experiment_name)
model_load_path=os.path.join(model_save_path,"epoch%d.pkl"%load_epoch_pkl)

# -----end set parameters----------

#def data_preparation(input_HR_path,processed_HR_path,GT_img_path,GT_npy_path,LR_path):
#    data=DataPrepare(input_HR_path,processed_HR_path,GT_img_path,GT_npy_path,LR_path,scale_factor,detectors_order,detectors)
#    data.dataPrepare()


def getDataLoader(LR_path,GT_path,isTrain):
    data = DriverDatasetSR(LR_path,GT_path,scale_factor,isTrain)
    dataloader = DataLoader(dataset=data,shuffle=True, batch_size=batch_size)
    return dataloader

def model_summary(model):
    summary(model, input_size=(3,HR_H,HR_W))

def getModel():
    if model_name=="SPResNet":
        return SPResNet(scale_factor=scale_factor,out_channels=3,num_ResBlock=num_ResBlock,feature_size=feature_size)
    else:
        raise NotImplementedError("model %s is not implemented"%model_name)


def start_train():
    train_dataloader=getDataLoader(train_LR_path,train_processed_HR_path,isTrain=True)
    valid_dataloader=getDataLoader(valid_LR_path,valid_processed_HR_path,isTrain=True)
    model=getModel()
    loss_func = F.mse_loss
    optimizer = optim.Adam(model.parameters())
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=step_size)
    
    trainer=Trainer(model,train_dataloader,valid_dataloader,loss_func,optimizer,lr_scheduler,max_epoch,log_path,model_save_path)
    print("start training %s."%experiment_name)
    trainer.train()
    print("finish training %s."%experiment_name)

def loadModel():
    print("loading model from %s"%model_load_path)
    location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_dict=torch.load(model_load_path,map_location=location)
    test_model=getModel()
    test_model.load_state_dict(state_dict)
    return test_model

def start_test():
    print("start testing.")
    test_model=loadModel()
    test_dataloader=getDataLoader(test_LR_path,test_processed_HR_path,isTrain=False) 
    loss_func = F.mse_loss
    tester=Tester(test_model,test_dataloader,test_output_path,detectors_order,loss_func)
    
    tester.test()
    print("finish testing.")

#def start_visualize():
#    b=Baseline(test_LR_path,baseline_output_path,scale_factor,detectors_order)
#    b.detect()
#    o=OutputVisualize(test_processed_HR_path,test_GT_img_path,test_output_path,baseline_output_path,visualize_output_path,threshold,detectors_order)
#    o.visualize()






if __name__ == '__main__':
    if isPrepare:
        data_preparation(valid_input_HR_path,valid_processed_HR_path,valid_GT_img_path,valid_GT_npy_path,valid_LR_path)
        data_preparation(train_input_HR_path,train_processed_HR_path,train_GT_img_path,train_GT_npy_path,train_LR_path)
    if isPrepare and isTest:
        data_preparation(test_input_HR_path,test_processed_HR_path,test_GT_img_path,test_GT_npy_path,test_LR_path)
    
    if isSummary:
        model_summary(getModel())

    if isTrain:
        start_train()

    if isTest:
        start_test()

    


    print("tensorboard cmd= tensorboard --logdir=%s"%log_root)
    


    









