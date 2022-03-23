from lib2to3.pgen2 import driver
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import config as cf
import math
import pymongo
import argparse
import random
import models
from models import SARSA
from models import Deep_Q_Network, net
import pandas as pd
from pymongo import MongoClient
from pandas import DataFrame,Series
import matplotlib.pyplot as plt, time
from matplotlib.patches import Circle
import pickle
import copy
from random import sample
import json
import os
os.environ["CUDA_DEVICES_ORDER"]="PCI_BUS_IS"
os.environ["CUDA_VISIBLE_DEVICES"]="5"

filename = "dict.json"

parser = argparse.ArgumentParser(description='Reinforce Learning')
#=======================================================================================================================
# Environment Parameters
parser.add_argument('--random_seed', default=19, type=int, help='The specific seed to generate the random numbers')
parser.add_argument('--numDrones', default=4, type=int, help='The number of Drones(UAV)')
parser.add_argument('--numUsers', default=1050, type=int, help='The number of Users')
parser.add_argument('--length', default=100, type=int, help='The length of the area(meter)')
parser.add_argument('--width', default=100, type=int, help='The width of the area(meter)')
parser.add_argument('--resolution', default=10, type=int, help='The Resolution (meter) for drones')
parser.add_argument('--episode', default=100, type=int, help='The number turns it plays')
parser.add_argument('--step', default=2000, type=int, help='The number of steps for any turn of runs')
parser.add_argument('--round', default=100, type=int, help='The number of rounds per training')
parser.add_argument('--interval', default=200, type=int, help='The interval between each chunk of training rounds')
parser.add_argument('--action_space', default=['east','west','south','north','stay'], type=list, help='The avaliable states')
parser.add_argument('--EPSILON', default=0.9, type=float, help='The greedy policy')
parser.add_argument('--ALPHA', default=0.1, type=float, help='The learning rate')
parser.add_argument('--LAMBDA', default=0.5, type=float, help='The discount factor')
parser.add_argument('--store_step', default=100, type=int, help='number of steps per storation, store the data from target network')
#=======================================================================================================================
# DQN Parameters
parser.add_argument('--lr', default=0.005, type=float, help='The learning rate for CNN')
parser.add_argument('--drop_rate', default=0.5, type=float, help='The drop out rate for CNN')
parser.add_argument('--iteration', default=1, type=int, help='The number of data per train')
parser.add_argument('--sequence_len', default=10, type=int, help='The number of observations in a sequence')
#=======================================================================================================================
# Signal Attenuation Parameters
parser.add_argument('--connectThresh', default=40, type=int, help='Threshold')
parser.add_argument('--dAngle', default=60, type=int, help='The directivity angle')
parser.add_argument('--fc', default=2.4e9, type=int, help='The carrier frequency')
parser.add_argument('--Pt', default=0, type=int, help='The drone transmit power in Watts')
parser.add_argument('--BW', default=200e3, type=int, help='The bandwidth')
parser.add_argument('--N0', default=10**(-20.4), type=float, help='The N0')
parser.add_argument('--SIGMA', default=20, type=int, help='The SIGMA')
#=======================================================================================================================
# Database Parameters
parser.add_argument('--database_name', default='DQN_Data_Base', type=str, help='The name of database')
parser.add_argument('--collection_name', default='Q_table_collection', type=str, help='The name of the collection')
parser.add_argument('--host', default='localhost', type=str, help='The host type')
parser.add_argument('--mongodb_port', default=27017, type=int, help='The port of database')

args = parser.parse_args()
sarsa = SARSA(args)
DQN = Deep_Q_Network(args)


def generate_pre_Q_dict_from_array(array):
    dict = {}
    for i in range(np.shape(array)[0]):
        data = '( '
        for j in range(np.shape(array)[1]):
            if j != 0:
                data += ', '
            data += str(array[i,j])
        data += ' )'
        dict [str(args.action_space[i])] = data
    return copy.deepcopy(dict)

def generate_dict_from_array(array, name):
    dict = {}
    for i in range(np.shape(array)[0]):
        data = '( '
        for j in range(np.shape(array)[1]):
            if j != 0:
                data += ', '
            data += str(array[i,j])
        data += ' )'
        dict [name + ' ' + str(i)] = data
    return copy.deepcopy(dict)

def environment(distribution, u):
    np.random.seed(args.random_seed)
    length = 100
    cluster = {}
    number = {}
    colour = {}
    SIGMA = {}
    label = {}
    # font = {'family': 'Palatino',}
    font = {}
    u['cluster1'][0] = (u['cluster1'][0] + random.randint(-15,15)) % length
    u['cluster1'][1] = (u['cluster1'][1] + random.randint(-15,15)) % length
    SIGMA['cluster1'] = 7
    number['cluster1'] = 250
    colour['cluster1'] = '#9400D3'
    cluster['cluster1'] = np.zeros((number['cluster1'], 3))
    cluster['cluster1'][:, 0:1] = np.floor(
        (distribution['cluster1x'] * SIGMA['cluster1'] + u['cluster1'][0]) % length)
    cluster['cluster1'][:, 1:2] = np.floor(
        (distribution['cluster1y'] * SIGMA['cluster1'] + u['cluster1'][1]) % length)
    label['cluster1'] = 'MS cluster 1'

    u['cluster2'][0] = (u['cluster2'][0] + random.randint(-15,15)) % length
    u['cluster2'][1] = (u['cluster2'][1] + random.randint(-15,15)) % length
    SIGMA['cluster2'] = 10
    number['cluster2'] = 300
    colour['cluster2'] = '#FF8C00'
    cluster['cluster2'] = np.zeros((number['cluster2'], 3))
    cluster['cluster2'][:, 0:1] = np.floor(
        (distribution['cluster2x'] * SIGMA['cluster2'] + u['cluster2'][0]) % length)
    cluster['cluster2'][:, 1:2] = np.floor(
        (distribution['cluster2y'] * SIGMA['cluster2'] + u['cluster2'][1]) % length)
    label['cluster2'] = 'MS cluster 2'

    #u['cluster3'][0] = (u['cluster3'][0] + random.randint(-15,15)) % length
    #u['cluster3'][1] = (u['cluster3'][1] + random.randint(-15,15)) % length
    SIGMA['cluster3'] = 6
    number['cluster3'] = 200
    colour['cluster3'] = '#228B22'
    cluster['cluster3'] = np.zeros((number['cluster3'], 3))
    cluster['cluster3'][:, 0:1] = np.floor(
        (distribution['cluster3x'] * SIGMA['cluster3'] + u['cluster3'][0]) % length)
    cluster['cluster3'][:, 1:2] = np.floor(
        (distribution['cluster3y'] * SIGMA['cluster3'] + u['cluster3'][1]) % length)
    label['cluster3'] = 'MS cluster 3'

    number['uniform'] = 300
    cluster['uniform'] = np.random.randint(0, 100, size=[number['uniform'], 3])
    colour['uniform'] = '#4169E1'
    label['uniform'] = 'MS uniform'

    userPos = cluster['cluster1']
    for dict in cluster:
        if dict != 'cluster1':
            userPos = np.concatenate((userPos, cluster[dict]), axis=0)
    userPos[:, 2] = 1.5
    #save_initial_settling(userPos,dronePos)
    return userPos

def environment_setup(i):
    np.random.seed(args.random_seed)
    u = np.random.randint(300,700)
    
    if args.numDrones == 2:
        dronePos = np.array([[5, 5, 30], [95, 95, 30]])
        #dronePos = np.array([[5, 5, 30], [5, 5, 30]])
    elif args.numDrones == 4:
        #dronePos = np.array([[5, 5, 30], [5, 95, 30], [95, 5, 30], [95, 95, 30]])
        #dronePos = np.array([[5, 5, 30], [5, 5, 30], [95, 95, 30], [95, 95, 30]])
        dronePos = np.array([[5, 5, 30], [5, 5, 30], [5, 5, 30], [5, 5, 30]])
    elif args.numDrones == 8:
        dronePos = np.array([[5, 5, 30], [95, 95, 30], [5, 95, 30], [95, 5, 30], [5, 50, 30], [50, 5, 30], [95, 50, 30], [50, 95, 30]])
    
    resolution = 10
    length = 100
    distribution = {}
    cluster = {}
    number = {}
    colour = {}
    SIGMA = {}
    u = {}
    label = {}
    # font = {'family': 'Palatino',}
    font = {}
    u['cluster1'] = [np.random.randint(20, 80), np.random.randint(20, 80)]
    SIGMA['cluster1'] = 7
    number['cluster1'] = 250
    colour['cluster1'] = '#9400D3'
    cluster['cluster1'] = np.zeros((number['cluster1'], 3))
    distribution['cluster1x'] = np.random.randn(number['cluster1'], 1)
    distribution['cluster1y'] = np.random.randn(number['cluster1'], 1)
    cluster['cluster1'][:, 0:1] = np.floor(
        (distribution['cluster1x'] * SIGMA['cluster1'] + u['cluster1'][0]) % length)
    cluster['cluster1'][:, 1:2] = np.floor(
        (distribution['cluster1y'] * SIGMA['cluster1'] + u['cluster1'][1]) % length)
    label['cluster1'] = 'MS cluster 1'

    u['cluster2'] = [np.random.randint(30, 80), np.random.randint(20, 70)]
    SIGMA['cluster2'] = 10
    number['cluster2'] = 300
    colour['cluster2'] = '#FF8C00'
    cluster['cluster2'] = np.zeros((number['cluster2'], 3))
    distribution['cluster2x'] = np.random.randn(number['cluster2'], 1)
    distribution['cluster2y'] = np.random.randn(number['cluster2'], 1)
    cluster['cluster2'][:, 0:1] = np.floor(
        (distribution['cluster2x'] * SIGMA['cluster2'] + u['cluster2'][0]) % length)
    cluster['cluster2'][:, 1:2] = np.floor(
        (distribution['cluster2y'] * SIGMA['cluster2'] + u['cluster2'][1]) % length)
    label['cluster2'] = 'MS cluster 2'

    u['cluster3'] = [np.random.randint(10, 85), np.random.randint(10, 90)]
    SIGMA['cluster3'] = 6
    number['cluster3'] = 200
    colour['cluster3'] = '#228B22'
    cluster['cluster3'] = np.zeros((number['cluster3'], 3))
    distribution['cluster3x'] = np.random.randn(number['cluster3'], 1)
    distribution['cluster3y'] = np.random.randn(number['cluster3'], 1)
    cluster['cluster3'][:, 0:1] = np.floor(
        (distribution['cluster3x'] * SIGMA['cluster3'] + u['cluster3'][0]) % length)
    cluster['cluster3'][:, 1:2] = np.floor(
        (distribution['cluster3y'] * SIGMA['cluster3'] + u['cluster3'][1]) % length)
    label['cluster3'] = 'MS cluster 3'

    number['uniform'] = 300
    cluster['uniform'] = np.random.randint(0, 100, size=[number['uniform'], 3])
    colour['uniform'] = '#4169E1'
    label['uniform'] = 'MS uniform'

    userPos = cluster['cluster1']
    for dict in cluster:
        if dict != 'cluster1':
            userPos = np.concatenate((userPos, cluster[dict]), axis=0)
    userPos[:, 2] = 1.5
    #save_initial_settling(userPos,dronePos)
    return dronePos, userPos, distribution, u


def refreash_dataset(name = args.database_name, collection_name = args.collection_name, host='localhost', port=27017):
    mongo_client = MongoClient(host, port) # 创建 MongoClient 对象，（string格式，int格式）
    mongo_db = mongo_client[name] # MongoDB 中可存在多个数据库，根据数据库名称获取数据库对象（Database）
    #mongo_db.authenticate(mongodb_user, mongodb_passwd) # 登录认证
    for collection in mongo_db.collection_names():
        print ('collection: ', collection, ' have been refreshed')
        db_collection=mongo_db[collection] # 每个数据库包含多个集合，根据集合名称获取集合对象（Collection）
        # drop = db_collection.drop() delate
        remove = db_collection.remove()
        #drop = db_collection.drop()


def save_initial_settling(U_p, D_p, name = args.database_name, collection_name ='initial_setting', host='localhost', port=27017):
    myclient = pymongo.MongoClient(host='localhost', port=27017)
    mydb = myclient[name]
    dblist = myclient.list_database_names()
    collection = mydb[collection_name]
    initial_info = {}
    initial_info ['random_seed'] = args.random_seed
    initial_info ['num_drones'] = args.numDrones
    initial_info ['num_users'] = args.numUsers
    initial_info ['user_positions'] = generate_dict_from_array(U_p, 'user')
    initial_info ['drone_positions'] = generate_dict_from_array(D_p, 'drone')
    initial_info ['carrier_frequency'] = args.fc
    initial_info ['transmit_power'] = args.Pt
    initial_info ['sinr_threshold'] = args.connectThresh
    initial_info ['drone_user_capacity'] = 'not consider yet'
    initial_info ['x_min'] = 0
    initial_info ['x_max'] = args.width
    initial_info ['y_min'] = 0
    initial_info ['y_max'] = args.length
    initial_info ['possible_actions'] = [[1,0],[-1,0],[0,1],[0,-1],[0,0]]
    initial_info ['learning_rate'] = args.ALPHA
    initial_info ['total_episodes'] = args.episode
    initial_info ['iterations_per_episode'] = args.step
    initial_info ['discount_factor'] = args.LAMBDA
    initial_info ['episodes'] = 'total if possible'
    result = collection.insert(initial_info)

def save_predicted_Q_table(observation_seq, SINR, predicted_table, action, reward_, dronePos, episode, step, drone, name , collection_name, host='localhost', port=27017):
    myclient = pymongo.MongoClient(host='localhost', port=27017)
    mydb = myclient[name]
    dblist = myclient.list_database_names()
    data = {}
    data['episode']=episode
    data['step'] = step
    data['drone number']=drone
    drone_dict = data ['qtable'] = {}
    drone_dict['position: (' + str(dronePos[int(drone),0])+', '+str(dronePos[int(drone),1])+')'] = {}
    drone_dict['position: (' + str(dronePos[int(drone),0])+', '+str(dronePos[int(drone),1])+')'] = generate_pre_Q_dict_from_array(predicted_table.T)
    drone_dict['SINR'] = generate_dict_from_array( SINR, 'user')
    drone_dict['state'] = generate_dict_from_array(dronePos, 'drone')
    drone_dict['action'] = action
    drone_dict['reward'] = reward_
    collection = mydb[collection_name]
    result = collection.insert(data)
    #print(result)

def save_data_for_training(Store_transition, count, observation_seq_adjust, action_adjust, reward_, observation_seq_adjust_):
    Store_transition[count%100] = {}
    # Store_transition[count]['observation_seq'] = np.array([observation_seq_adjust])
    Store_transition[count%100]['observation_seq'] = observation_seq_adjust
    Store_transition[count%100]['action'] = action_adjust
    Store_transition[count%100]['reward_'] = reward_
    Store_transition[count%100]['observation_seq_'] = observation_seq_adjust_
    # np.save('Data\\' + str(count - count%args.store_step ) + '_to_' + str(count - count%args.store_step + args.store_step - 1) + '.npy', Store_transition)
    return Store_transition

def grasp_data_for_training(Store_transition, count, eval_network, target_network, numbers = 10):
    r_ = []
    k = 0
    action_ = []
    Store = Store_transition
    if count<100:
        selected = sample([i for i in range(count)], numbers)
    else:
        selected = sample([i for i in range(100)], numbers)
    for dict in selected:
        state = Store[dict]['observation_seq']
        state_ = Store[dict]['observation_seq_']
        r = Store[dict]['reward_']
        action = Store[dict]['action']
        state = torch.from_numpy(np.array([(np.swapaxes(np.swapaxes(state,0,2),1,2)).astype(np.float32)]))
        state_ = torch.from_numpy(np.array([(np.swapaxes(np.swapaxes(state_,0,2),1,2)).astype(np.float32)]))
        if k == 0:
            Q_eval = torch.unsqueeze(eval_network(state.cuda())[0][action],0)
            Q_next = target_network(state_.cuda())
        else:
            Q_eval = torch.cat([Q_eval,torch.unsqueeze(eval_network(state.cuda())[0][action],0)],dim=0)
            Q_next = torch.cat([Q_next,target_network(state_.cuda())],dim=0)
        r_ = np.append(r_,r/1050.0)
        action_ = np.append(action_,action)
        k = k + 1
    return  Q_eval, r_, action_, Q_next


def main(args):
    # ========================================== start up eval net =====================================================
    eval_network = []
    param_eval = []
    optimizer_eval = []
    param_target = []
    optimizer_target = []
    target_network = []
    dcounts = []
    Lambda = 0.5
    gama = []
    TotalMax = []
    TotalMin = []
    EachMax = []
    EachMin = []
    for i in range(args.numDrones):
        dcounts.append([])
        EachMax.append([])
        EachMin.append([])
        eval_network.append(net(args))
        if cf.use_cuda:
            eval_network[i].cuda()
        cudnn.benchmark = True
        param_eval.append(list(eval_network[i].parameters()))
        optimizer_eval.append(optim.SGD(param_eval[i], lr=args.lr, momentum=0.9, weight_decay=1e-3))
    # ========================================== start up target net ===================================================
        target_network.append(net(args))
        if cf.use_cuda:
            target_network[i].cuda()
        cudnn.benchmark = True
        param_target.append(list(target_network[i].parameters()))
        optimizer_target.append(optim.SGD(param_target[i], lr=args.lr, momentum=0.9, weight_decay=1e-3))
    # =============================================== start up =========================================================
    counts = []
    dronePos, userPos, distribution, u = environment_setup(0)
    for i in range(args.episode):
        #userPos = environment(distribution, u)
        #if i % 15 == 0 and i != 0:
        #    userPos = environment(distribution, u)
        MaxReward = 0
        EachMaxReward = []
        MinReward = 1050
        EachMinReward = []
        total = 0
        dtotal = []
        counter = 0
        Store_transition = []
        count = []
        for drone_No in range(args.numDrones):
            Store_transition.append({})
            count.append(0)
            dtotal.append(0)
            EachMaxReward.append(0)
            EachMinReward.append(0)
        for j in range(9):
            for drone_No in range(args.numDrones):
                allocVec, SINR, reward = models.alloc_users(userPos,dronePos,args.fc,args.dAngle,args.N0,args.BW,args.Pt,args.connectThresh)
                observation_seq = DQN.observe(drone_No, allocVec['total'], dronePos, userPos)
                observation_seq_adjust = (np.swapaxes(np.swapaxes(observation_seq,0,2),1,2)).astype(np.float32) # too meet the need of torch input
                if cf.use_cuda:
                    action_reward = target_network[drone_No](torch.from_numpy(np.array([observation_seq_adjust])).cuda())
                    action_reward = action_reward.cpu()
                else:
                    action_reward = target_network[drone_No](torch.from_numpy(np.array([observation_seq_adjust])))
            # ================================ greedy actions ======================================================
                if random.random() < args.EPSILON:
                    action_adjust = (torch.argmax(action_reward)).detach().numpy()
                else:
                    action_adjust = np.array(sample([i for i in range(len(args.action_space))], 1)[0])
                # ================================= take actions =======================================================
                dronePos[drone_No][:2] = DQN.take_action(dronePos[drone_No][:2], args.action_space[action_adjust])
                allocVec_, SINR_, reward_ = models.alloc_users(userPos,dronePos,args.fc,args.dAngle,args.N0,args.BW,args.Pt,args.connectThresh)
                observation_seq_ = DQN.observe(drone_No, allocVec_['total'], dronePos, userPos)
                observation_seq_adjust_ = (np.swapaxes(np.swapaxes(observation_seq_,0,2),1,2)).astype(np.float32)                  
                Store_transition[drone_No] = save_data_for_training(Store_transition[drone_No], count[drone_No], observation_seq, action_adjust, reward_['total'], observation_seq_)
                count[drone_No] += 1
        for j in range(args.step):
            for drone_No in range(args.numDrones):
                allocVec, SINR, reward = models.alloc_users(userPos,dronePos,args.fc,args.dAngle,args.N0,args.BW,args.Pt,args.connectThresh)
                observation_seq = DQN.observe(drone_No, allocVec['total'], dronePos, userPos)
                observation_seq_adjust = (np.swapaxes(np.swapaxes(observation_seq,0,2),1,2)).astype(np.float32) # too meet the need of torch input
                if cf.use_cuda:
                    action_reward = target_network[drone_No](torch.from_numpy(np.array([observation_seq_adjust])).cuda())
                    action_reward = action_reward.cpu()
                else:
                    action_reward = target_network[drone_No](torch.from_numpy(np.array([observation_seq_adjust])))
            # ================================ greedy actions ======================================================
                if random.random() < args.EPSILON:
                    action_adjust = (torch.argmax(action_reward)).detach().numpy()
                else:
                    action_adjust = np.array(sample([i for i in range(len(args.action_space))], 1)[0])
                # ================================= take actions =======================================================
                dronePos[drone_No][:2] = DQN.take_action(dronePos[drone_No][:2], args.action_space[action_adjust])
                allocVec_, SINR_, reward_ = models.alloc_users(userPos,dronePos,args.fc,args.dAngle,args.N0,args.BW,args.Pt,args.connectThresh)
                observation_seq_ = DQN.observe(drone_No, allocVec_['total'], dronePos, userPos)
                observation_seq_adjust_ = (np.swapaxes(np.swapaxes(observation_seq_,0,2),1,2)).astype(np.float32)                      
                Store_transition[drone_No] = save_data_for_training(Store_transition[drone_No], count[drone_No], observation_seq, action_adjust, reward_['total'], observation_seq_)
                count[drone_No] += 1
                #save_predicted_Q_table(observation_seq, SINR, action_reward.detach().numpy(), args.action_space[action_adjust], reward_, dronePos, i, j, drone_No, args.database_name, args.collection_name)
                Q_eval, re, action, Q_next = grasp_data_for_training(Store_transition[drone_No], count[drone_No], eval_network[drone_No], target_network[drone_No])
                loss = DQN.pred_loss(torch.from_numpy(re.astype(np.float32)).cuda(), Q_next, Q_eval, Lambda)
                optimizer_eval[drone_No].zero_grad()  
                loss.backward(retain_graph=True)
                optimizer_eval[drone_No].step()  
                if count[drone_No] % args.interval == 0 :          
                    torch.save(eval_network[drone_No].state_dict(), 'Network Parameters\\' + str(drone_No) + 'th_eval_network_parameters')
                    File = open(filename, 'r')
                    DroneDict = json.load(File)
                    if DroneDict['Pause_Drone'][str(drone_No)] == False:
                        target_network[drone_No].load_state_dict(torch.load('Network Parameters\\' + str(drone_No) + 'th_eval_network_parameters'))
                        print('Drone' + str(drone_No) + ' update')
                    print ('Network Parameters\\' + str(drone_No) + 'th_eval_network_parameters is successfully load to the target network')
            total += reward_['total']
            counter += 1
            if MaxReward < total/counter:
                MaxReward = total/counter
            if MinReward > total/counter:
                MinReward = total/counter
            for drone_No in range(args.numDrones):
                dtotal[drone_No] += reward_[str(drone_No)]
                if EachMaxReward[drone_No] < dtotal[drone_No]/counter:
                    EachMaxReward[drone_No] = dtotal[drone_No]/counter
                if EachMinReward[drone_No] > dtotal[drone_No]/counter:
                    EachMinReward[drone_No] = dtotal[drone_No]/counter
            if j%20 ==0:
                print('episode', i,' with average reward:', total/counter)
                for drone_No in range(args.numDrones):
                    print('episode', str(i),'drone' + str(drone_No) + ' with average reward:', dtotal[drone_No]/counter)
        counts += [total / counter]
        TotalMax += [MaxReward]
        TotalMin += [MinReward]
        print('All episodes rewards:', counts)
        for drone_No in range(args.numDrones):
            EachMin[drone_No] += [EachMinReward[drone_No]]
            EachMax[drone_No] += [EachMaxReward[drone_No]]
            dcounts[drone_No] += [dtotal[drone_No] / counter]
            np.save('drone' + str(drone_No) +'_episode_99.npy', dcounts[drone_No])
            print('drone' + str(drone_No) + ' rewards:', dcounts[drone_No])
        np.save('reward.npy', counts)
        np.save('TotalMax.npy',TotalMax)
        np.save('TotalMin.npy',TotalMin)
        np.save('EachMin.npy',EachMin)
        np.save('EachMax.npy',EachMax)
    np.save('gama.npy',gama)
if __name__ == "__main__":
    main(args)