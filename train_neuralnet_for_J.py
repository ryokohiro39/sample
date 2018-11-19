# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import numpy as np
from two_layer_net_for_J import TwoLayerNet_for_J

from common.functions import *
from common.optimizer import *
# データの読み込み
#x_train = np.random.randint(0,4,(2511,18))
#t_train = np.random.randint(0,4,(2511,2))
x_train = np.loadtxt('x_train.csv', delimiter=',', skiprows=1)#2009-2017
t_train = np.loadtxt('t_train.csv', delimiter=',', skiprows=1)#2009-2017
#print(x_train)
network = TwoLayerNet_for_J(input_size=18, hidden_size=10, output_size=2)

iters_num = 1000 
train_size = x_train.shape[0]
batch_size = 31*3
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)#1674/93=18

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    #print(batch_mask)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 勾配
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    #print(grad)
    
    # 更新
    optimizer = Adam()
    params = network.params
    optimizer.update(params,grad)
        
    loss = mean_squared_error(network.predict(x_batch), t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        #print(network.predict(x_batch),t_batch)	
        #print(loss)
        #print(x_batch)
        pass

x_test = np.loadtxt('x_test.csv', delimiter=',', skiprows=1)#2018
t_test = np.loadtxt('t_test.csv', delimiter=',', skiprows=1)#2018

predict_test = network.predict(x_test)
loss_t = mean_squared_error(predict_test, t_test) / x_test.shape[0]
correct = 0
draw = 0
homewin = 0
awaywin =0
match = 0
bias = 0.5

for i in range(261):
	if predict_test[i,1] + bias < predict_test[i,0]:
		win_p = 0
		match += 1
		if t_test[i,1] < t_test[i,0]:
			win_t = 0
			homewin += 1
		elif t_test[i,0] < t_test[i,1]:
			win_t = 1
			awaywin += 1
		else:
			win_t = 2
			draw += 1
		if win_p == win_t:
			correct += 1
	elif predict_test[i,0] + bias < predict_test[i,1]:
		win_p = 1
		match += 1
		if t_test[i,1] < t_test[i,0]:
			win_t = 0
			homewin += 1
		elif t_test[i,0] < t_test[i,1]:
			win_t = 1
			awaywin += 1
		else:
			win_t = 2
			draw += 1
		if win_p == win_t:
			correct += 1
	else:
		win_p = 3
		
#match = x_test.shape[0]
accuracy = correct / match
homewin_rate = homewin / match
awaywin_rate = awaywin / match
draw_rate = draw / match
#print(predict, t_test)
print('test result')
print('loss:%s'%(loss_t))
print('match:%s'%(match))
print('accuracy:%s'%(accuracy))
print('homewin_rate:%s'%(homewin_rate))
print('awaywin_rate:%s'%(awaywin_rate))
print('draw_rate:%s\n'%(draw_rate))

predict_train = network.predict(x_train)
loss_t = mean_squared_error(predict_train, t_train) / x_train.shape[0]
correct = 0
draw = 0
homewin = 0
awaywin =0
match = 0
bias = 0.5

for i in range(1674):
	if predict_train[i,1] + bias < predict_train[i,0]:
		win_p = 0
		match += 1
		if t_train[i,1] < t_train[i,0]:
			win_t = 0
			homewin += 1
		elif t_train[i,0] < t_train[i,1]:
			win_t = 1
			awaywin += 1
		else:
			win_t = 2
			draw += 1
		if win_p == win_t:
			correct += 1
	elif predict_train[i,0] + bias < predict_train[i,1]:
		win_p = 1
		match += 1
		if t_train[i,1] < t_train[i,0]:
			win_t = 0
			homewin += 1
		elif t_train[i,0] < t_train[i,1]:
			win_t = 1
			awaywin += 1
		else:
			win_t = 2
			draw += 1
		if win_p == win_t:
			correct += 1
	else:
		win_p = 3
		
#match = x_train.shape[0]
accuracy = correct / match
homewin_rate = homewin / match
awaywin_rate = awaywin / match
draw_rate = draw / match
#print(predict, t_train)
print('train result')
print('loss:%s'%(loss_t))
print('match:%s'%(match))
print('accuracy:%s'%(accuracy))
print('homewin_rate:%s'%(homewin_rate))
print('awaywin_rate:%s'%(awaywin_rate))
print('draw_rate:%s\n'%(draw_rate))