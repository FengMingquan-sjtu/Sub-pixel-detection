#torch modules
import torch.nn as nn
from torchvision import transforms as T
import torch
from torch.utils.tensorboard import SummaryWriter

#python modules
import os
from PIL import Image


class Trainer():
    def __init__(self,model,train_dataloader,valid_dataloader,loss_func,optimizer,lr_scheduler,max_epoch,log_path,model_save_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model=model.to(self.device)

        self.dataloader={"train":train_dataloader,"valid":valid_dataloader}
        self.data_size={"train":len(train_dataloader),"valid":len(valid_dataloader)}

        self.loss_func=loss_func
        self.optimizer=optimizer
        self.lr_scheduler=lr_scheduler
        self.max_epoch=max_epoch

        self.writer=SummaryWriter(log_path)

        self.model_save_path=model_save_path
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)



    def train(self):
        for epoch in range(self.max_epoch):
            for phase in ["train","valid"]:
                torch.set_grad_enabled(phase == 'train')
                if phase=="train":
                    self.model.train()
                else:
                    self.model.eval()
                epoch_loss_sum=0
                for lr,gt in self.dataloader[phase]:
                    lr=lr.to(self.device) #low resolution img
                    gt=gt.to(self.device) #ground truth
                    self.optimizer.zero_grad()
                    output=self.model(lr)
                    loss=self.loss_func(output,gt)
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()#update weight

                epoch_loss_sum+=loss.item() * gt.size(0)
                if phase == 'train':
                    self.lr_scheduler.step()#update learning rate 
            

                epoch_loss=epoch_loss_sum/self.data_size[phase]

                #output and log statistics
                print("epoch=%d,phase=%s,loss=%f"%(epoch+1,phase,epoch_loss))
                self.writer.add_scalar('%s_loss'%phase,epoch_loss,(epoch+1)*self.data_size['train'])
            self._save_model(epoch=epoch+1)
        self.writer.close()

    def _save_model(self,epoch):
        pkl_path=os.path.join(self.model_save_path,"epoch%d.pkl"%epoch)
        torch.save(self.model.state_dict(),pkl_path)



class Tester():
    def __init__(self,model):
        self.model=model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model=model.to(self.device)
        self.toPIL=T.Compose([T.ToPILImage(),])
        

    def test(self,test_dataloader,test_output_path):
        if not os.path.exists(test_output_path):
            os.makedirs(test_output_path)

        self.model.eval()
        with torch.no_grad():
            for lr,name in test_dataloader:
                lr=lr.to(self.device)
                output=self.model(lr)
                for idx,img in enumerate(output.cpu().data):
                    path=os.path.join(test_output_path,name[idx])
                    img=self.toPIL(img)
                    img.save(path)






    