import os
import torch
from torch.utils import data
import torch.nn.functional as F
import torch
from config import Config
import numpy as np
import torch.nn as nn
from model import Backbone_rg,Arcface,FocalLoss,cal_accuracy,ArcMarginProduct
from torch.nn import DataParallel
from data.dataset import *
from tqdm import tqdm
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
import argparse
import os
import argparse
from torch.nn.parameter import Parameter

parser = argparse.ArgumentParser()
parser.add_argument("--target", type=str, default="human",help="human or animal")
parser.add_argument("--mode", type=str,default="full", help="eye or noise or full")
parser.add_argument("--lr", type=float,default=1e-1, help="eye or noise or full")
parser.add_argument("--rf", type=int,default=0, help="reget fc weight")
parser.add_argument("--tfc", type=int,default=1, help="train fc or not")
args = parser.parse_args()

opt = Config()
opt.batch_size = 512
opt.lr = args.lr
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
opt.save_epoch=1
opt.max_epoch = 20
if args.target=="human":
    num_class = 9833
else:
    num_class = 407

def get_weightdata(model,dataloader,num_class):
    ##get class center
    
    new_weight = torch.zeros(num_class, 512)
    count = np.zeros(num_class)
    pbar = tqdm(dataloader)
    model.eval()
    for data_input, label in pbar:
        data_input = data_input.to(opt.device)
        feature = model(data_input).data.cpu().numpy()
        for i in range(len(label)):
            new_weight[label[i]]+=feature[i]
            count[label[i]]+=1
    for i in range(len(count)):
        new_weight[i] /=count[i]
    return new_weight.t()


if __name__ == '__main__':

    if args.target=="human":
        train_dataset = Dataset('CASIA-maxpy-clean_crop', 'datalist/casia_trian.txt',args.mode)
    else:
        train_dataset = Dataset('Pig', 'datalist/train.txt',args.mode)
        val_dataset = TestLoader('Pig', 'datalist/test.txt',args.mode)
    
    training_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=4,
                                      batch_size=opt.batch_size,
                                      shuffle=True)
    if not args.target=="human":
        validation_data_loader = torch.utils.data.DataLoader(dataset=val_dataset, num_workers=1,
                                        batch_size=opt.batch_size//2,
                                        shuffle=False)
    #criterion = FocalLoss(gamma=2)
    criterion = nn.CrossEntropyLoss()
    model = Backbone_rg(num_layers=50, drop_ratio=0.5, mode='ir_se')
    model.to(opt.device)
    model = DataParallel(model)
    if args.target=="human":
        model.load_state_dict(torch.load('weight/arc3_50.pth'))
    elif args.mode=="eye":
        model.load_state_dict(torch.load('weight/human_eye.pth'))
    elif args.mode=="full":
        model.load_state_dict(torch.load('weight/human_full.pth'))
    model.to(opt.device)
    weight_file_name = "weight/"+args.target+"_"+args.mode+"_data.npy"
    if args.rf:
        fc_weight = get_weightdata(model,training_data_loader,num_class)
        np.save(weight_file_name,fc_weight.data.cpu().numpy())
    fc_weight = np.load(weight_file_name)
    fc_weight = torch.from_numpy(fc_weight)
    #metric_fc = Arcface(embedding_size=512, classnum=fc_weight.shape[0])
    #metric_fc = nn.Linear(512,num_class)
    metric_fc = ArcMarginProduct(512, num_class, s=30, m=0.5)
    metric_fc.weight = Parameter(fc_weight.transpose(0,1))
    metric_fc.to(opt.device)
    metric_fc = DataParallel(metric_fc)
    if args.tfc:
        optimizer = torch.optim.SGD([{'params': model.parameters()},{'params': metric_fc.parameters()}],lr=opt.lr, momentum = 0.9,weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.SGD([{'params': model.parameters()}],lr=opt.lr, momentum = 0.9,weight_decay=opt.weight_decay)
    
    scheduler = MultiStepLR(optimizer, milestones=[15,18], gamma=0.1)

    log = open("log"+args.target+"_"+args.mode,'w+')
    for i in range(opt.max_epoch):
        iters = 0
        model.train()
        pbar = tqdm(training_data_loader)
        for data_input, label in pbar:
            data_input = data_input.to(opt.device)
            label = label.to(opt.device).long()
            feature = model(data_input)
            output = metric_fc(feature,label)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            label = label.data.cpu().numpy()
            acc = np.mean((output == label).astype(int))
            pbar.set_description(str(loss.data.cpu())+' '+str(acc))
            if iters%100==0:
                log.write(str(loss.data.cpu())+' '+str(acc)+"\n")
            iters+=1
        if i==opt.max_epoch-1:
            torch.save(model.state_dict(),'weight/'+args.target+"_"+args.mode+'.pth')
        scheduler.step()

        ####eval
        if not args.target=="human":
            test_pbar = tqdm(validation_data_loader,total=len(validation_data_loader))

            sims,labels = [],[]
            for data1,data2 ,label in test_pbar:
                data1 = data1.to(opt.device)
                data2 = data2.to(opt.device)
                labels.extend(label)
                label = label.to(opt.device)
                feature1 = model(data1).view(-1,512)
                feature2 =  model(data2).view(-1,512)
                sim = torch.cosine_similarity(feature1, feature2, dim=1).data.cpu().numpy().tolist()
                sims.extend(sim)

            acc,th = cal_accuracy(sims, labels)
            log.write("eval:"+str(i)+" "+str(acc)+' '+str(th)+"\n")
