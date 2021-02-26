import sys
import pandas as pd
import numpy as np
import csv

def train_adagrad(x,y,feature):
    dim = feature * 9 + 1
    w = np.zeros([dim, 1])
    x = np.concatenate((np.ones([11 * 471, 1]), x), axis = 1).astype(float)
    
    learning_rate = 0.6
    iter_time = 5000
    adagrad = np.zeros([dim, 1])
    
    eps = 0.0000000001
    for t in range(iter_time):
        loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/471/12)#rmse
        if(t%100==0):
            print(str(t) + ":" + str(loss))
        for batch in range(11):
            gradient = 2 * np.dot(x[batch*471:(batch+1)*471].transpose(), np.dot(x[batch*471:(batch+1)*471], w) - y[batch*471:(batch+1)*471])
            adagrad += gradient ** 2
            w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
    np.save('weight_best.npy', w)

def train_adam(x,y,feature_num):
    dim = feature_num * 9 + 1
    w = np.random.rand(dim, 1)
    x = np.concatenate((np.ones([11 * 471, 1]), x), axis = 1).astype(float)

    learning_rate = 0.05
    iter_time = 700
    mt =  np.zeros([dim, 1])
    vt =  np.zeros([dim, 1])
    beta1 = 0.9
    beta2 = 0.999
    eps = 0.00000001
    for t in range(1,iter_time+1):
        loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/471/12)#rmse
        if(t%100==0):
            print(str(t) + ":" + str(loss))
        for batch in range(11):
            gradient = 2 * np.dot(x[batch*471:(batch+1)*471].transpose(), np.dot(x[batch*471:(batch+1)*471], w) - y[batch*471:(batch+1)*471])
        ##gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y) #dim*1
            mt = beta1*mt+(1-beta1)*gradient
            vt = beta2*vt+(1-beta2)*(gradient**2)
            mt_n = mt/(1-beta1**t)
            vt_n = vt/(1-beta2**t)
            w = w - learning_rate * mt_n /(np.sqrt(vt_n)+eps)
    np.save('weight_best.npy', w)

def test(feature_num,feature_neglect):
    testdata = pd.read_csv(sys.argv[2], header = None, encoding = 'big5')

    test_data = testdata.iloc[:, 2:]
    test_data[test_data == 'NR'] = 0
    test_data[test_data == '-1'] = 0
    test_data = test_data.to_numpy()
    test_x = np.empty([240, (feature_num)*9], dtype = float)
   
    for i in range(240):
        count = 0
        for feat in range(18):
            if feat%18 not in feature_neglect:
                if feat%18 in {14}:
                    temp = np.array(test_data[18 * i+feat, :].reshape(1, -1),dtype=float)
                    #print(temp)
                    test_x[i, 9*(feature_num-4):9*(feature_num-3)] = np.cos(temp*np.pi/180)
                    test_x[i, 9*(feature_num-3):9*(feature_num-2)] = np.sin(temp*np.pi/180)
                elif feat%18 in {15}:
                    temp = np.array(test_data[18 * i+feat, :].reshape(1, -1),dtype=float)
                    #print(temp)
                    test_x[i, 9*(feature_num-2):9*(feature_num-1)] = np.cos(temp*np.pi/180)
                    test_x[i, 9*(feature_num-1):9*feature_num] = np.sin(temp*np.pi/180)
                else:
                    test_x[i, 9*count:9*(count+1)] = test_data[18 * i+feat, :].reshape(1, -1)
                    count += 1
    std_x = np.load('std_x_best.npy')
    mean_x = np.load('mean_x_best.npy')
    for i in range(len(test_x)):
        for j in range(len(test_x[0])):
            if std_x[j] != 0:
                test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
    test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)

    w = np.load('weight_best.npy')
    ans_y = np.dot(test_x, w)
    
    with open(sys.argv[3], mode='w', newline='') as submit_file:
        csv_writer = csv.writer(submit_file)
        header = ['id', 'value']
        print(header)
        csv_writer.writerow(header)
        for i in range(240):
            row = ['id_' + str(i), ans_y[i][0]]
            csv_writer.writerow(row)
            print(row)

if __name__ == '__main__':
    
    filename = sys.argv[0]
    if(sys.argv[1])=='-test':
        testingfile = sys.argv[2]
        outputfile = sys.argv[3]
        feature_neglect = {0,1,2,3,4,12,13}
        feature_num = 18-len(feature_neglect)+2
    else:
        trainingfile = sys.argv[1]
        testingfile = sys.argv[2]
        outputfile = sys.argv[3]
        ##Preprocessing
        data = pd.read_csv(trainingfile, encoding = 'big5')
        data = data.iloc[:, 3:]
        data[data == 'NR'] = 0
        data[data == '-1'] = 0
        raw_data = data.to_numpy()
        ##Extract Features1 & Preprocessing
        month_data = {}
        ######
        feature_neglect = {0,1,2,3,4,12,13}
        feature_num = 18-len(feature_neglect)+4
        ######
        feature_pm25 = 9
        feature_winddir = 14
        for item in feature_neglect:
            if int(item)<9:
                feature_pm25 -=1
            if int(item)<14:
                feature_winddir -=1
        for month in range(12):
            if(month==6): continue
            sample = np.empty([feature_num, 480])
            for day in range(20):
                count = 0
                feature_add = 0     
                for i in range(18):
                    if i not in feature_neglect:
                        sample[count, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day)+i, :]
                        count += 1
                
            sample[18-len(feature_neglect)]=np.cos(sample[feature_winddir]*np.pi/180)
            sample[18-len(feature_neglect)+1]=np.sin(sample[feature_winddir]*np.pi/180)
            sample[18-len(feature_neglect)+2]=np.cos(sample[feature_winddir+1]*np.pi/180) 
            sample[18-len(feature_neglect)+3]=np.sin(sample[feature_winddir+1]*np.pi/180)
            
            sample = np.delete(sample, feature_winddir, 0)
            sample = np.delete(sample, feature_winddir, 0)
            
            if month >6:
                month_data[month-1] = sample
            else:
                month_data[month] = sample
        
        feature_num -=2
        ##Extract Features2
        x = np.empty([11 * 471, feature_num * 9], dtype = float) 
        y = np.empty([11 * 471, 1], dtype = float)
        for month in range(11):
            for day in range(20):
                for hour in range(24):
                    if day == 19 and hour > 14:
                        continue
                    x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) #vector dim:10*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
                    ### fix
                    y[month * 471 + day * 24 + hour, 0] = month_data[month][feature_pm25, day * 24 + hour + 9] #value
                    ###
        print(x)
        for i in range(18):
            print(x[i][:8])
        print(y)
        
        ##Normalize1
        mean_x = np.mean(x, axis = 0) #18 * 9 
        std_x = np.std(x, axis = 0) #18 * 9 
        for i in range(len(x)): #12 * 471
            for j in range(len(x[0])): #18 * 9 
                if std_x[j] != 0:
                    x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
        np.save('mean_x_best.npy', mean_x)
        np.save('std_x_best.npy', std_x)
        import math
        x_train_set = x[: math.floor(len(x) * 0.8), :]
        y_train_set = y[: math.floor(len(y) * 0.8), :]
        x_validation = x[math.floor(len(x) * 0.8): , :]
        y_validation = y[math.floor(len(y) * 0.8): , :]
        print(x_train_set.shape)
        print(y_train_set)
        print(x_validation.shape)
        print(y_validation)
        print(len(x_train_set))
        print(len(y_train_set)) 
        print(len(x_validation))
        print(len(y_validation))
        train_adagrad(x,y,feature_num)
        #train_adam(x,y,feature_num)
    test(feature_num,feature_neglect)
   
