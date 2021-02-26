import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F

def load_training_data(path='training_label.txt'):
    # 把 training 時需要的 data 讀進來
    # 如果是 'training_label.txt'，需要讀取 label，如果是 'training_nolabel.txt'，不需要讀取 label
    if 'training_label' in path:
        with open(path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip('\n').replace(' x',' ').replace(' x',' ').replace(' x',' ').split(' ') for line in lines]
        x = [line[2:] for line in lines]
        y = [line[0] for line in lines]
        for i in range(len(x)):
            x[i] = [chrs.replace('ed ',' ') for chrs in x[i]]
            x[i] = [chrs.replace('es ',' ') for chrs in x[i]]
            x[i] = [chrs.replace('s ',' ') for chrs in x[i]]
            stop_words =[" i ", " me ", " my ", " myself", " we", " our", " ours", " ourselves ", " you ", " your ", " yours ", " yourself ", " yourselves ", " he ", " him ", " his ", " himself ", " she ", " her ", " hers ", " herself ", " it ", " its ", " itself ", " they ", " them ", " their ", " theirs ", " themselves ", " what ", " which ", " who ", " whom ", " this ", " that ", " these ", " those ", " am ", " is ", " are ", " was ", " were ", " be ", " been ", " being ", " have ", " has ", " had ", " having ", " do ", " does ", " did ", " doing ", " a ", " an ", " the ", " and ", " but ", " if ", " or ", " because ", " as ", " until ", " while ", " of ", " at ", " by ", " for ", " with ", " about ", " against ", " between ", " into ", " through ", " during ", " before ", " after ", " above ", " below ", " to ", " from ", " up ", " down ", " in ", " out ", " on ", " off ", " over ", " under ", " again ", " further ", " then ", " once ", " here ", " there ", " when ", " where ", " why ", " how ", " all ", " any ", " both ", " each ", " few ", " more ", " most ", " other ", " some ", " such ", " no ", " nor ", " not ", " only ", " own ", " same ", " so ", " than ", " too ", " very ", " s ", " t ", " can ", " will ", " just ", " don ", " should ", " now "]
            for j in stop_words :
                x[i] = [chrs.replace(j,' ') for chrs in x[i]]

        return x, y
    else:
        with open(path, 'r') as f:
            lines = f.readlines()
            x = [line.strip('\n').replace(' x' ,' ').replace(' x',' ').replace(' x',' ').split(' ') for line in lines]
            for i in range(len(x)):
                x[i] = [chrs.replace('ed ',' ') for chrs in x[i]]
                x[i] = [chrs.replace('es ',' ') for chrs in x[i]]
                x[i] = [chrs.replace('s ',' ') for chrs in x[i]]
                stop_words =[" i ", " me ", " my ", " myself", " we", " our", " ours", " ourselves ", " you ", " your ", " yours ", " yourself ", " yourselves ", " he ", " him ", " his ", " himself ", " she ", " her ", " hers ", " herself ", " it ", " its ", " itself ", " they ", " them ", " their ", " theirs ", " themselves ", " what ", " which ", " who ", " whom ", " this ", " that ", " these ", " those ", " am ", " is ", " are ", " was ", " were ", " be ", " been ", " being ", " have ", " has ", " had ", " having ", " do ", " does ", " did ", " doing ", " a ", " an ", " the ", " and ", " but ", " if ", " or ", " because ", " as ", " until ", " while ", " of ", " at ", " by ", " for ", " with ", " about ", " against ", " between ", " into ", " through ", " during ", " before ", " after ", " above ", " below ", " to ", " from ", " up ", " down ", " in ", " out ", " on ", " off ", " over ", " under ", " again ", " further ", " then ", " once ", " here ", " there ", " when ", " where ", " why ", " how ", " all ", " any ", " both ", " each ", " few ", " more ", " most ", " other ", " some ", " such ", " no ", " nor ", " not ", " only ", " own ", " same ", " so ", " than ", " too ", " very ", " s ", " t ", " can ", " will ", " just ", " don ", " should ", " now "]
                for j in stop_words :
                    x[i] = [chrs.replace(j,' ') for chrs in x[i]]

        return x

def load_testing_data(path='testing_data'):
    # 把 testing 時需要的 data 讀進來
    with open(path, 'r') as f:
        lines = f.readlines()
        X = ["".join(line.strip('\n').replace(' x',' ').replace(' x',' ').replace(' x',' ').split(",")[1:]).strip() for line in lines[1:]]
        X = [sen.split(' ') for sen in X]
        for i in range(len(X)):
            X[i] = [chrs.replace('ed ',' ') for chrs in X[i]]
            X[i] = [chrs.replace('es ',' ') for chrs in X[i]]
            X[i] = [chrs.replace('s ',' ') for chrs in X[i]]
            stop_words =[" i ", " me ", " my ", " myself", " we", " our", " ours", " ourselves ", " you ", " your ", " yours ", " yourself ", " yourselves ", " he ", " him ", " his ", " himself ", " she ", " her ", " hers ", " herself ", " it ", " its ", " itself ", " they ", " them ", " their ", " theirs ", " themselves ", " what ", " which ", " who ", " whom ", " this ", " that ", " these ", " those ", " am ", " is ", " are ", " was ", " were ", " be ", " been ", " being ", " have ", " has ", " had ", " having ", " do ", " does ", " did ", " doing ", " a ", " an ", " the ", " and ", " but ", " if ", " or ", " because ", " as ", " until ", " while ", " of ", " at ", " by ", " for ", " with ", " about ", " against ", " between ", " into ", " through ", " during ", " before ", " after ", " above ", " below ", " to ", " from ", " up ", " down ", " in ", " out ", " on ", " off ", " over ", " under ", " again ", " further ", " then ", " once ", " here ", " there ", " when ", " where ", " why ", " how ", " all ", " any ", " both ", " each ", " few ", " more ", " most ", " other ", " some ", " such ", " no ", " nor ", " not ", " only ", " own ", " same ", " so ", " than ", " too ", " very ", " s ", " t ", " can ", " will ", " just ", " don ", " should ", " now "]
            for j in stop_words :
                X[i] = [chrs.replace(j,' ') for chrs in X[i]]

    return X

def evaluation(outputs, labels):
    # outputs => probability (float)
    # labels => labels
    outputs[outputs>=0.5] = 1 # 大於等於 0.5 為有惡意
    outputs[outputs<0.5] = 0 # 小於 0.5 為無惡意
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct