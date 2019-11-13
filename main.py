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
from utils.driverDataset import DriverDataset
from utils.dataPrepare import DataPrepare,cv2_interestDetect
from utils.option import getArgs
from models.model import SPResNet

# -----begin set parameters----------
args=getArgs()
batch_size=args.batch_size
max_epoch=args.max_epoch
step_size=args.step_size #learning rate update step
scale_factor=args.scale_factor #scale_factor must be a factor of HR_H and HR_W
feature_size=args.feature_size
num_ResBlock=args.num_ResBlock
max_corners=args.max_corners
threshold=args.threshold
model_name=args.model_name
experiment_name=args.experiment_name
load_epoch_pkl=args.load_epoch_pkl
isTrain=args.isTrain
isTest=args.isTest
isPrepare=args.isPrepare
isSummary=args.isSummary


HR_height=1200
HR_width=2000

data_root="data"
valid_input_HR_path=os.path.join(data_root,"valid_input")
valid_GT_path=os.path.join(data_root,"valid_GT")
valid_LR_path=os.path.join(data_root,"valid_LR")
train_input_HR_path=os.path.join(data_root,"train_input")
train_GT_path=os.path.join(data_root,"train_GT")
train_LR_path=os.path.join(data_root,"train_LR")
test_input_HR_path=os.path.join(data_root,"test_input")
test_GT_path=os.path.join(data_root,"test_GT")
test_LR_path=os.path.join(data_root,"test_LR")
test_output_path=os.path.join(data_root,"test_output")

log_root="logs"
log_path=os.path.join("logs",experiment_name)
model_save_path=os.path.join("weight",experiment_name)
model_load_path=os.path.join(model_save_path,"epoch%d.pkl"%load_epoch_pkl)

# -----end set parameters----------

def data_preparation(input_HR_path,GT_path,LR_path):
    interestPointsDetect=cv2_interestDetect
    data=DataPrepare(input_HR_path,GT_path,LR_path,scale_factor,interestPointsDetect,max_corners)
    data.dataPrepare()


def getDataLoader(LR_path,GT_path):
    data = DriverDataset(LR_path,GT_path,HR_height,HR_width,scale_factor)
    dataloader = DataLoader(dataset=data,shuffle=True, batch_size=batch_size)
    return dataloader

def model_summary(model):
    in_H=HR_height//scale_factor
    in_W=HR_width//scale_factor 
    summary(model, input_size=(3,in_H,in_W))

def getModel():
    if model_name=="SPResNet":
        return SPResNet(scale_factor=scale_factor,num_ResBlock=num_ResBlock,feature_size=feature_size)
    else:
        raise NotImplementedError("model %s is not implemented"%model_name)


def start_train():
    train_dataloader=getDataLoader(train_LR_path,train_GT_path)
    valid_dataloader=getDataLoader(valid_LR_path,valid_GT_path)
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
    state_dict=torch.load(model_load_path)
    test_model=getModel()
    test_model.load_state_dict(state_dict)
    return test_model

def start_test():
    print("start testing.")
    test_model=loadModel()
    tester=Tester(test_model)
    test_dataloader=getDataLoader(test_LR_path,None) # "None" means GT imgs are invisible to tester
    tester.test(test_dataloader,test_output_path)
    print("finish testing.")





if __name__ == '__main__':
    if isPrepare:
        data_preparation(valid_input_HR_path,valid_GT_path,valid_LR_path)
        data_preparation(train_input_HR_path,train_GT_path,train_LR_path)
    if isPrepare and isTest:
        data_preparation(test_input_HR_path,test_GT_path,test_LR_path)
    
    if isSummary:
        model_summary(getModel())

    if isTrain:
        start_train()

    if isTest:
        start_test()

    print("tensorboard cmd= tensorboard --logdir=%s"%log_root)
    


    









