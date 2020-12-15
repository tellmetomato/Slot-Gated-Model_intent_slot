import argparse
import time
import os
import numpy as np
from collections import Counter
from torch.optim.lr_scheduler import *

from builddataset import build_dataset
from atisdata import ATISData 
from SlotFillingAndIntentDetermination import SlotFillingAndIntentDetermination
#from Transformer import Transformer
#from Bertmodel import Bertmodel
import math
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score


def train(train_data_path, test_data_path, mode, bidirectional, saved_model_path, cuda, all_data_path):
    train_X, train_y ,train_y_2,i_t,i_s,i_in = build_dataset(train_data_path,all_data_path)
    train_set = ATISData(train_X, train_y,train_y_2)
    train_loader = DataLoader(dataset=train_set,batch_size=1,shuffle=True)
    vocab_size = i_t    #词典大小 16850 108 37
    label_size = i_s    #槽标签总数 108
    intent_size = i_in   #意图总数 38
    print(vocab_size,label_size,intent_size)

    model = SlotFillingAndIntentDetermination(vocab_size = vocab_size, label_size = label_size, mode=mode, bidirectional=bidirectional,intent_size = intent_size)

    if cuda:
        model = model.cuda()
    loss_fn_1 = nn.CrossEntropyLoss()
    loss_fn_2 = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    scheduler=StepLR(optimizer,step_size=3)
    epoch_num = 20
    print_step = 10000
    model.train()
    for epoch in range(epoch_num):
        start_time = time.time()
        running_loss = 0.0
        count = 0
        scheduler.step()
        for X, y,y2 in train_loader:
            optimizer.zero_grad()
            if torch.__version__ < "0.4.*":
                X, y ,y2= Variable(X), Variable(y),Variable(y2)
            if cuda:
                X, y ,y2= X.cuda(), y.cuda(),y2.cuda()
            model.train()
            output,output2 = model(X)
            output = output.squeeze(0)
            y = y.squeeze(0)
            sentence_len = y.size(0)

            loss1 = loss_fn_1(output, y.long())
            loss2 = loss_fn_2(output2,y2.long())
            loss = 0.6*loss2 + 0.4*loss1
            loss.backward()

            optimizer.step()
            if torch.__version__ < "0.4.*":
                running_loss += loss.data[0] / sentence_len
            else:
                running_loss += ((loss.item() / sentence_len) if sentence_len != 0 else 0)
            count += 1
            if count % print_step == 0 :
                print("epoch: %d, loss: %.4f" % (epoch, running_loss / print_step))
                running_loss = 0.0
                count = 0
        print("time: ", time.time() - start_time)
        do_eval(model, train_loader, cuda)
    torch.save(model.state_dict(), saved_model_path)


def predict(train_data_path, test_data_path, mode, bidirectional, saved_model_path, result_path, cuda, all_data_path):
    test_X, test_y,test_y_2 ,i_t,i_s,i_in= build_dataset(test_data_path, all_data_path)

    test_set = ATISData(test_X, test_y,test_y_2)
    test_loader = DataLoader(dataset=test_set, batch_size=1,shuffle=False)
    vocab_size = i_t    #词典大小
    label_size = i_s    #槽标签总数
    intent_size = i_in    #意图总数 14546 108 38


    model = SlotFillingAndIntentDetermination(vocab_size = vocab_size, label_size = label_size, mode=mode, bidirectional=bidirectional,intent_size = intent_size)

    model.load_state_dict(torch.load(saved_model_path))
    model.eval()
    if cuda:
        model = model.cuda()
    do_eval(model, test_loader, cuda)


def accuracy(predictions, labels):     #计算准确率
    return (100.0 * np.sum(np.array(predictions) == np.array(labels)) / len(labels))

def F(predictions, labels):
    return [f1_score(labels,predictions, average='micro')*100,f1_score(labels,predictions, average='macro')*100]

def do_eval(model, test_loader, cuda):
    model.eval()
    predicted = []
    predicted2 = []
    true_label = []
    true_intent = []
    for X, y ,y2 in test_loader:
        X = Variable(X)
        if cuda:
            X = X.cuda()
        output,output2 = model(X)
        output = output.squeeze(0)
        
        _, output = torch.max(output, 1)
        _, output2 = torch.max(output2,1)
        if cuda:
            output = output.cpu()
            output2 = output2.cpu()
        predicted.extend(output.data.numpy().tolist())
        predicted2.extend(output2.data.numpy().tolist())
        y = y.squeeze(0)
        
        true_label.extend(y.numpy().tolist())
        true_intent.extend(y2.numpy().tolist())
    print("F1: %.3f" % F(predicted, true_label)[0],"+++++++++++",F(predicted, true_label)[1])
    print("Acc-: %.3f" % accuracy(predicted2, true_intent))
    return predicted,predicted2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, default="./data/train.txt")
    parser.add_argument('--test_data_path', type=str, default="./data/test.txt")
    parser.add_argument('--all_data_path', type=str, default="./data/all_data.txt")
    parser.add_argument('--saved_model_path', type=str, default="./saved_models/model.model")
    parser.add_argument('--result_path', type=str, default="./data/output.txt")
    parser.add_argument('--mode', type=str, default='gru' )
    parser.add_argument('--bidirectional', action='store_true', default=True)   
    parser.add_argument('--cuda', type=bool, default=True)
    args = parser.parse_args()
    if os.path.exists(args.saved_model_path):
        print("predicting...")
        predict(args.train_data_path, args.test_data_path, 
                args.mode, args.bidirectional, args.saved_model_path, args.result_path, args.cuda, args.all_data_path)
    else:
        print("training")
        train(args.train_data_path, args.test_data_path, 
                args.mode, args.bidirectional, args.saved_model_path, args.cuda, args.all_data_path)