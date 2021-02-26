import numpy as np
import matplotlib.pyplot as plt
import sys

def _normalize(X, train = True, specified_column = None, X_mean = None, X_std = None):
    # This function normalizes specific columns of X.
    # The mean and standard variance of training data will be reused when processing testing data.
    #
    # Arguments:
    #     X: data to be processed
    #     train: 'True' when processing training data, 'False' for testing data
    #     specific_column: indexes of the columns that will be normalized. If 'None', all columns
    #         will be normalized.
    #     X_mean: mean value of training data, used when train = 'False'
    #     X_std: standard deviation of training data, used when train = 'False'
    # Outputs:
    #     X: normalized data
    #     X_mean: computed mean value of training data
    #     X_std: computed standard deviation of training data

    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column] ,0).reshape(1, -1)
        X_std  = np.std(X[:, specified_column], 0).reshape(1, -1)

    X[:,specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)
     
    return X, X_mean, X_std

def _train_dev_split(X, Y, dev_ratio = 0.25):
    # This function spilts data into training set and development set.
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]


def _shuffle(X, Y):
    # This function shuffles two equal-length list/array, X and Y, together.
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def _sigmoid(z):
    # Sigmoid function can be used to calculate probability.
    # To avoid overflow, minimum/maximum output value is set.
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))

def _f(X, w, b):
    # This is the logistic regression function, parameterized by w and b
    #
    # Arguements:
    #     X: input data, shape = [batch_size, data_dimension]
    #     w: weight vector, shape = [data_dimension, ]
    #     b: bias, scalar
    # Output:
    #     predicted probability of each row of X being positively labeled, shape = [batch_size, ]
    return _sigmoid(np.matmul(X, w) + b)

def _predict(X, w, b):
    # This function returns a truth value prediction for each row of X 
    # by rounding the result of logistic regression function.
    return np.round(_f(X, w, b)).astype(np.int)
    
def _accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc

def _cross_entropy_loss(y_pred, Y_label):
    # This function computes the cross entropy.
    #
    # Arguements:
    #     y_pred: probabilistic predictions, float vector
    #     Y_label: ground truth labels, bool vector
    # Output:
    #     cross entropy, scalar
    cross_entropy = -np.dot(Y_label, np.log(y_pred)) - np.dot((1 - Y_label), np.log(1 - y_pred))
    return cross_entropy

def _gradient(X, Y_label, w, b):
    # This function computes the gradient of cross entropy loss with respect to weight w and bias b.
    y_pred = _f(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.sum(pred_error * X.T, 1)
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad

def append_features(X_train,X_test,handle_index):
    X_train = X_train.T
    X_test = X_test.T
    for i in handle_index:
        for _exp in [0.2,0.4,0.6,0.8,1.2]:
            X_sq = X_train[i]**_exp
            X_train = np.insert(X_train,len(X_train),X_sq, axis =0) 
            X_sqt = X_test[i]**_exp
            X_test = np.insert(X_test,len(X_test),X_sqt, axis =0)
        #print(X_train[i+handle][3:9])
    X_train = X_train.T
    X_test = X_test.T
    return(X_train,X_test)

def delete_features(X_train,X_test,delete_index):
    X_train = X_train.T
    X_test = X_test.T
    for i in range(129):#364
        X_train = np.delete(X_train,364, axis =0) 
        X_test = np.delete(X_test,364, axis =0)
        #print(X_train[i+handle][3:9])
    count = 0
    for i in delete_index:#364
        X_train = np.delete(X_train,i-count, axis =0) 
        X_test = np.delete(X_test,i-count, axis =0)
        count+=1
    """
    for i in range(51):#225
        X_train = np.delete(X_train,225, axis =0) 
        X_test = np.delete(X_test,225, axis =0)
    """
    X_train = X_train.T
    X_test = X_test.T
    return(X_train,X_test)

def train(X_train,Y_train,max_iter,batch_size,learning_rate,_random):
    dev_ratio = 0.1
    X_train=X_train
    X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio = dev_ratio)

    train_size = X_train.shape[0]
    dev_size = X_dev.shape[0]
    test_size = X_test.shape[0]
    data_dim = X_train.shape[1]
    print('Size of training set: {}'.format(train_size))
    print('Size of development set: {}'.format(dev_size))
    print('Size of testing set: {}'.format(test_size))
    print('Dimension of data: {}'.format(data_dim))
    # Zero initialization for weights ans bias
    if(_random==True):
        w = np.random.rand(data_dim)
        b = np.random.rand(1) 
    else:
        w = np.zeros((data_dim,)) 
        b = np.zeros((1,))
    print(w.shape)
    print(b.shape)
    # Keep the loss and accuracy at every iteration for plotting
    train_loss = []
    dev_loss = []
    train_acc = []
    dev_acc = []
    # Calcuate the number of parameter updates
    step = 1
    adagrad_w = 0
    adagrad_b = 0
    eps = 0.000001
    # Iterative training
    for epoch in range(max_iter):
        # Random shuffle at the begging of each epoch
        X_train, Y_train = _shuffle(X_train, Y_train)
            
        # Mini-batch training
        for idx in range(int(np.floor(train_size / batch_size))):
            X = X_train[idx*batch_size:(idx+1)*batch_size]
            Y = Y_train[idx*batch_size:(idx+1)*batch_size]

            # Compute the gradient
            w_grad, b_grad = _gradient(X, Y, w, b)
            adagrad_w += (w_grad**2)  
            adagrad_b += (b_grad**2)    
            # gradient descent update
            # learning rate decay with time
            #w = w - learning_rate/np.sqrt(adagrad_w + eps) * w_grad
            #b = b - learning_rate/np.sqrt(adagrad_b + eps) * b_grad
            w = w - learning_rate/np.sqrt(step) * w_grad
            b = b - learning_rate/np.sqrt(step) * b_grad

            step = step + 1
                
        # Compute loss and accuracy of training set and development set
        y_train_pred = _f(X_train, w, b)
        Y_train_pred = np.round(y_train_pred)
        train_acc.append(_accuracy(Y_train_pred, Y_train))
        train_loss.append(_cross_entropy_loss(y_train_pred, Y_train) / train_size)

        y_dev_pred = _f(X_dev, w, b)
        Y_dev_pred = np.round(y_dev_pred)
        dev_acc.append(_accuracy(Y_dev_pred, Y_dev))
        dev_loss.append(_cross_entropy_loss(y_dev_pred, Y_dev) / dev_size)

    #print(w)
    #print(b)
    print('Training loss: {}'.format(train_loss[-1]))
    print('Development loss: {}'.format(dev_loss[-1]))
    print('Training accuracy: {}'.format(train_acc[-1]))
    print('Development accuracy: {}'.format(dev_acc[-1]))

    # Loss curve
    
    plt.plot(train_loss)
    plt.plot(dev_loss)
    plt.title('Loss')
    plt.legend(['train', 'dev'])
    plt.savefig('loss.png')
    #plt.show()

    # Accuracy curve
    plt.plot(train_acc)
    plt.plot(dev_acc)
    plt.title('Accuracy')
    plt.legend(['train', 'dev'])
    plt.savefig('acc.png')
    #plt.show()
    print(w.shape)
    print(b.shape)
    
    return(w,b)

def predict(X_test,w,b):
    predictions = _predict(X_test, w, b)
    return predictions

if __name__ == '__main__':
    #python hw2_best -train X_train Y_train X_test Output
    #python hw2_best -test X_train Y_train X_test Output w1 b1 w2 b2 
    np.random.seed(0)
    _simple = False
    if(len(sys.argv)<=2):
        print("ERROR")
        print("USAGE:")
        print("python hw2_best -train X_train Y_train X_test Output")
        print("python hw2_best -test X_train Y_train X_test Output w1 b1 w2 b2")
        exit()
    if(sys.argv[1]=="-train"):
        print("train")
        do_train = True
    elif(sys.argv[1]=="-test"):
        print("test only")
        do_train = False
        wens_fpath = sys.argv[16]
        bens_fpath = sys.argv[17]
    else:
        print("ERROR")
        print("USAGE:")
        print("python hw2_best -train X_train Y_train X_test Output")
        print("python hw2_best -test X_train Y_train X_test Output w1 b1 w2 b2")
        exit()
    if(sys.argv[2]=='-simple'):
        _simple = True
        X_train_fpath = sys.argv[3]
        Y_train_fpath = sys.argv[4]
        X_test_fpath = sys.argv[5]
        output_fpath = sys.argv[6]
    else:
        X_train_fpath = sys.argv[2]
        Y_train_fpath = sys.argv[3]
        X_test_fpath = sys.argv[4]
        output_fpath = sys.argv[5]
        w1_fpath = sys.argv[6]
        b1_fpath = sys.argv[7]
        w2_fpath = sys.argv[8]
        b2_fpath = sys.argv[9]
        w3_fpath = sys.argv[10]
        b3_fpath = sys.argv[11]
        w4_fpath = sys.argv[12]
        b4_fpath = sys.argv[13]
        w5_fpath = sys.argv[14]
        b5_fpath = sys.argv[15]

    # Parse csv files to numpy array

    with open(X_train_fpath) as f:
        next(f)
        X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
    with open(Y_train_fpath) as f:
        next(f)
        Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
    with open(X_test_fpath) as f:
        next(f)
        X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
    
    # Preprocessing 

    insert_feature_num = [0,126,210,211,212,507] #[0,126,210,211,212,507]
    X_train,X_test = append_features(X_train,X_test,insert_feature_num)
    #delete_feature_num = [1, 17, 54, 131, 178, 203, 242, 250, 327, 331, 337, 340, 346, 350, 351, 352, 354, 355, 358]
    delete_feature_num =[338,339,340,341,342,343,344,345,346,347,348,349,350,354,355,356,357]
    X_train,X_test = delete_features(X_train,X_test,delete_feature_num)

    
    # Normalize training and testing data
    X_train, X_mean, X_std = _normalize(X_train, train = True)
    X_test, _, _= _normalize(X_test, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)

    # Train
    if(do_train):
        if(_simple):
            w,b = train(X_train,Y_train,100,8,0.2,False)
            predictions = predict(X_test,w,b)
        else:    
            np.random.seed(20200322)
            w,b = train(X_train,Y_train,100,8,0.2,True)
            print("Save Seed 1")
            np.save(w1_fpath, w)
            np.save(b1_fpath, b)
            np.random.seed(19990523)
            w,b = train(X_train,Y_train,100,8,0.2,True)
            print("Save Seed 2")
            np.save(w2_fpath, w)
            np.save(b2_fpath, b)
            np.random.seed(12345678)
            w,b = train(X_train,Y_train,100,8,0.2,True)
            print("Save Seed 3")
            np.save(w3_fpath, w)
            np.save(b3_fpath, b)
            np.random.seed(18395909)
            w,b = train(X_train,Y_train,100,8,0.2,True)
            print("Save Seed 4")
            np.save(w4_fpath, w)
            np.save(b4_fpath, b)
            np.random.seed(19990426)
            w,b = train(X_train,Y_train,100,8,0.2,True)
            print("Save Seed 5")
            np.save(w5_fpath, w)
            np.save(b5_fpath, b)

    
        
    # Predict testing labels
    if(do_train == False):
        w1 = np.load(w1_fpath)
        b1 = np.load(b1_fpath)
        w2 = np.load(w2_fpath)
        b2 = np.load(b2_fpath)
        w3 = np.load(w3_fpath)
        b3 = np.load(b3_fpath)
        w4 = np.load(w4_fpath)
        b4 = np.load(b4_fpath)
        w5 = np.load(w5_fpath)
        b5 = np.load(b5_fpath)
        prediction = np.zeros((5,len(X_train)))
        prediction[0] = predict(X_train,w1,b1)
        prediction[1] = predict(X_train,w2,b2)
        prediction[2] = predict(X_train,w3,b3)
        prediction[3] = predict(X_train,w4,b4)
        prediction[4] = predict(X_train,w5,b5)
        prediction = prediction - 0.5
        #print(predictions.shape) 
        prediction = prediction.T
        print("Start Ensemble Training")
        w,b = train(prediction, Y_train,10,8,0.2,False)
        np.save(wens_fpath,w)
        np.save(bens_fpath,b)

        test = np.zeros((5,len(X_test)))
        test[0] = predict(X_test,w1,b1)
        test[1] = predict(X_test,w2,b2)
        test[2] = predict(X_test,w3,b3)
        test[3] = predict(X_test,w4,b4)
        test[4] = predict(X_test,w5,b5)
        test = test - 0.5
        #print(predictions.shape) 
        test = test.T
        predictions = predict(test,w,b)

    with open(output_fpath.format('logistic'), 'w') as f:
        f.write('id,label\n')
        for i, label in  enumerate(predictions):
            f.write('{},{}\n'.format(i, label))
    # Print out the most significant weights
    
    exit()
    ind = np.argsort(np.abs(w))[::-1]
    
    with open(X_test_fpath) as f:
        content = f.readline().strip('\n').split(',')
    features = np.array(content)
    for i in ind[0:10]:
        print(features[i], w[i])
