import sys
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

def train(x,y,lr,w):
    dim = 18 * 9 + 1
    #w = np.zeros([dim, 1])
    x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float)
    learning_rate = lr
    iter_time = 10000
    adagrad = np.zeros([dim, 1])
    eps = 0.0000000001
    loss_plt = []
    for t in range(iter_time):
        loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/471/12)#rmse
        loss_plt.append(loss)
        if(t%100==0):
            print(str(t) + ":" + str(loss))
        gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y) #dim*1
        adagrad += gradient ** 2
        w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
    np.save('weight.npy', w)
    return(loss_plt)

def test():
    testdata = pd.read_csv(sys.argv[2], header = None, encoding = 'big5')
    test_data = testdata.iloc[:, 2:]
    test_data[test_data == 'NR'] = 0
    test_data = test_data.to_numpy()
    test_x = np.empty([240, 18*9], dtype = float)
    for i in range(240):
        test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
    std_x = np.load('std_x.npy')
    mean_x = np.load('mean_x.npy')
    for i in range(len(test_x)):
        for j in range(len(test_x[0])):
            if std_x[j] != 0:
                test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
    test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)
    ##Prediction
    w = np.load('weight.npy')
    ans_y = np.dot(test_x, w)
    ##Save CSV
    with open(sys.argv[3], mode='w', newline='') as submit_file:
        csv_writer = csv.writer(submit_file)
        header = ['id', 'value']
        print(header)
        csv_writer.writerow(header)
        for i in range(240):
            row = ['id_' + str(i), ans_y[i][0]]
            csv_writer.writerow(row)
            print(row)

def print_lr(x,y):
    w = np.random.rand(18 * 9 + 1,1)
    iteration = []
    for i in range(1000):
        iteration.append(i+1)
    
    los_100 = train(x,y,0.5,w)
    print("10")
    los_10 = train(x,y,0.2,w)
    print("1")
    los_1 = train(x,y,0.1,w)
    print("0.1")
    los_01 = train(x,y,0.05,w)
    plt.plot(iteration,los_100,label="LR = 0.5")
    plt.plot(iteration,los_10,label="LR = 0.2")
    plt.plot(iteration,los_1,label="LR = 0.1")
    plt.plot(iteration,los_01,label="LR = 0.05")
    plt.legend(loc='upper right')
    plt.title("Loss to Iteration with Different LR", x=0.5, y=1.03)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()

def test_valid(x_train,y_train,x_valid,y_valid,lr,feature_num):
    x_train = np.concatenate((np.ones([4521, 1]), x_train), axis = 1).astype(float)
    w = np.zeros([feature_num*9+1, 1])
    learning_rate = lr
    iter_time = 10000
    adagrad = np.zeros([feature_num*9+1, 1])
    eps = 0.0000000001
    loss_plt = []
    for t in range(iter_time):
        loss = np.sqrt(np.sum(np.power(np.dot(x_train, w) - y_train, 2))/4521)#rmse
        loss_plt.append(loss)
        if(t==9999):
            print("")
            print("")
            if feature_num == 1:
                print("              Only PM 2.5")
            else: 
                print("              All Features")
            print("_________________________________________")
            print("| Train set loss at Iter 10000 | "+ str(round(loss, 5))+"|")
        gradient = 2 * np.dot(x_train.transpose(), np.dot(x_train, w) - y_train) #dim*1
        adagrad += gradient ** 2
        w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
    np.save('weight_val.npy', w)

    x_valid = np.concatenate(((np.ones([1131,1])),x_valid), axis = 1).astype(float)
    ans_y = np.dot(x_valid, w)
    loss = np.sqrt(np.sum(np.power(ans_y - y_valid, 2))/1131)
    print("| Validation set loss          | "+ str(round(loss, 5))+" |")
    print("_________________________________________")
    print("")
    

if __name__ == '__main__':
    filename = sys.argv[0]
    if(sys.argv[1])=='-test':
        testingfile = sys.argv[2]
        outputfile = sys.argv[3]
    else:
        trainingfile = sys.argv[1]
        testingfile = sys.argv[2]
        outputfile = sys.argv[3]
        ##Preprocessing
        data = pd.read_csv(trainingfile, encoding = 'big5')
        data = data.iloc[:, 3:]
        data[data == 'NR'] = 0
        raw_data = data.to_numpy()
        ##Extract Features1
        month_data = {}
        for month in range(12):
            sample = np.empty([18, 480])
            for day in range(20):
                sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
            month_data[month] = sample
        ##Extract Features2
        x = np.empty([12 * 471, 18 * 9], dtype = float)
        x_pm = np.empty([12*471,1*9], dtype=float)
        y = np.empty([12 * 471, 1], dtype = float)
        for month in range(12):
            for day in range(20):
                for hour in range(24):
                    if day == 19 and hour > 14:
                        continue
                    x_pm[month * 471 + day * 24 + hour, :] = month_data[month][9,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1)
                    x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
                    y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9] #value
        #print(x)
        #print(y)
        #print(x_pm)
        
        ##Normalize1
        
        mean_x = np.mean(x, axis = 0) #18 * 9 
        std_x = np.std(x, axis = 0) #18 * 9 
        for i in range(len(x)): #12 * 471
            for j in range(len(x[0])): #18 * 9 
                if std_x[j] != 0:
                    x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
        np.save('mean_x.npy', mean_x)
        np.save('std_x.npy', std_x)
        import math
        x_train_set = x[: math.floor(len(x) * 0.8), :]
        y_train_set = y[: math.floor(len(y) * 0.8), :]
        x_pm_train_set = x_pm[: math.floor(len(x_pm)*0.8), :]
        x_validation = x[math.floor(len(x) * 0.8): , :]
        y_validation = y[math.floor(len(y) * 0.8): , :]
        x_pm_validation = x_pm[math.floor(len(x_pm)*0.8):, :]
        """
        print(x_train_set.shape)
        print(y_train_set.shape)
        print(x_validation.shape)
        print(y_validation.shape)
        print(len(x_train_set))
        print(len(y_train_set))
        print(len(x_pm_train_set))
        print(len(x_validation))
        print(len(y_validation))
        print(len(x_pm_validation))
        """
        w = np.zeros([18 * 9 + 1, 1])
        train(x,y,100,w)
    test()
    #test_valid(x_train_set,y_train_set,x_validation,y_validation,100,18)
    #test_valid(x_pm_train_set,y_train_set,x_pm_validation,y_validation,100,1)
    