import sys
import torch
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from preprocess import MyDataset,get_dataloader,concatenate_dataloader
from train import loss_fn_kd,run_epoch
from model import StudentNet
from wq import encode8,encode16,decode8,decode16
import pickle

if __name__ == "__main__":

    if(len(sys.argv) < 2):
        exit()
    if(sys.argv[1]=="-help"):
        print("Usage: python3 hw7.py -train <option> <data directory>  <prediction file>")
        print("Usage: python3 hw7.py -test <testing data>  <prediction file>" )
        print("Usage: python3 hw7.py -wq <option> <origin model> <new model>")
        print("Train Option")
        print("-kd: Knowledge Distillation")
        print("-np: Network Pruning")
        
        exit()
    if(len(sys.argv) < 4):
        print("Usage: python3 hw7.py -train <option> <data directory>  <prediction file>")
        print("Usage: python3 hw7.py -test <testing data> <option> <model> <prediction file>" )
        print("Usage: python3 hw7.py -wq <option> <origin model> <new model>")
        #print("Usage: python3 main.py -ensemble_train <training label data>  <training unlabel data> <testing data>")
        #print("Usage: python3 main.py -ensemble_test <testing data>  <prediction file>")
        exit()
    if(sys.argv[1]=="-test"):
        testTransform = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])
        print("Loading test data...")
        test_dataloader = get_dataloader(sys.argv[2],testTransform,'testing', batch_size=32)
        decode_num = int(sys.argv[3])
        model = StudentNet(base=16).cuda()
        if(decode_num==8):
            state_dict = decode8(sys.argv[4])
        if(decode_num==16):
            state_dict = decode16(sys.argv[4])
        model.load_state_dict(state_dict)
        model.eval()
        prediction = []
        print("Start testing")
        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                inputs, hard_labels = data
                test_pred = model(inputs.cuda())
                test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            
                for y in test_label:
                    prediction.append(y)

        with open(sys.argv[5], 'w') as f:
            f.write('Id,label\n')
            for i, y in  enumerate(prediction):
                f.write('{},{}\n'.format(i, y))

    if(sys.argv[1]=="-wq"):
        params = torch.load(sys.argv[3])
        encode_num = int(sys.argv[2])
        if(encode_num==8):
            print("Encoding with 8 bits...")
            encode8(params, sys.argv[4])
            print(f"8-bit cost: {os.stat(sys.argv[4]).st_size} bytes.")
        if(encode_num==16):
            print("Encoding with 16 bits...")
            encode16(params, sys.argv[4])
            print(f"16-bit cost: {os.stat(sys.argv[4]).st_size} bytes.")
    
    if(sys.argv[1]=="-train"):
        print("Training...")
        trainTransform = transforms.Compose([
            transforms.RandomCrop(256, pad_if_needed=True, padding_mode='symmetric'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])
        testTransform = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])
        print("Loading train data...")
        # get dataloader
        train_dataloader = get_dataloader(sys.argv[3],trainTransform,'training', batch_size=32)
        print("Loading validation data...")
        valid_dataloader = get_dataloader(sys.argv[3],testTransform,'validation', batch_size=32)
        if(sys.argv[2]=="-kd"):
            
            teacher_net = models.resnet18(pretrained=False, num_classes=11).cuda()
            student_net = StudentNet(base=16).cuda()

            teacher_net.load_state_dict(torch.load(f'./teacher_resnet18.bin'))
            optimizer = optim.AdamW(student_net.parameters(), lr=1e-3)
            optimizer2 = optim.SGD(student_net.parameters(), lr=0.01, weight_decay=0.001)
            # TeacherNet永遠都是Eval mode.
            teacher_net.eval()
            now_best_acc = 0
            best_acc_epoch = 0
            total_epoch = 200
            change = False
            print("Start training...")
            for epoch in range(total_epoch):
                if(epoch>total_epoch/2):
                    if(not change):
                        print("Changing optimizer from Adam to SGD at epoch ",epoch)
                        change = True
                    student_net.train()
                    train_loss, train_acc = run_epoch(optimizer2,train_dataloader,teacher_net,student_net, update=True)
                    student_net.eval()
                    valid_loss, valid_acc = run_epoch(optimizer2,valid_dataloader,teacher_net,student_net, update=False)

                student_net.train()
                train_loss, train_acc = run_epoch(optimizer,train_dataloader,teacher_net,student_net, update=True)
                student_net.eval()
                valid_loss, valid_acc = run_epoch(optimizer,valid_dataloader,teacher_net,student_net, update=False)

                # 存下最好的model。
                if valid_acc > now_best_acc:
                    now_best_acc = valid_acc
                    torch.save(student_net.state_dict(), 'student_model.bin')
                    best_acc_epoch = epoch
                print('epoch {:>3d}: train loss: {:6.4f}, acc {:6.4f} valid loss: {:6.4f}, acc {:6.4f}'.format(
                    epoch, train_loss, train_acc, valid_loss, valid_acc))
            print("Best model saved, epoch = {best_epoch}, accuracy = {best_acc}".format(best_epoch=best_acc_epoch,best_acc=now_best_acc))
            
            print("Concatenating data...")
            train_dataloader = concatenate_dataloader(sys.argv[3],trainTransform,'training', batch_size=32)
            change = False
            student_net = StudentNet(base=16).cuda()
            optimizer = optim.AdamW(student_net.parameters(), lr=1e-3)
            optimizer2 = optim.SGD(student_net.parameters(), lr=0.01, weight_decay=0.001)
            for epoch in range(best_acc_epoch+1):
                if(epoch>total_epoch/2):
                    if(not change):
                        print("Changing optimizer from Adam to SGD at epoch ",epoch)
                        change = True
                    student_net.train()
                    train_loss, train_acc = run_epoch(optimizer2,train_dataloader,teacher_net,student_net, update=True)
        
                student_net.train()
                train_loss, train_acc = run_epoch(optimizer,train_dataloader,teacher_net,student_net, update=True)
                
                #torch.save(student_net.state_dict(), 'student_model.bin')
                print('epoch {:>3d}: train loss: {:6.4f}, acc {:6.4f}'.format(
                    epoch, train_loss, train_acc))
            torch.save(student_net.state_dict(), 'student_model_con.bin')
            print("Best model saved, epoch = {best_epoch}".format(best_epoch=best_acc_epoch))