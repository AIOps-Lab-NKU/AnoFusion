from generate_channels import get_channels
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize
from sklearn.metrics import mutual_info_score
import sys
sys.path.append('../')
sys.path.append('../model')

from AnoFusion import Net
from MyDataset import MyTorchDataset
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import argparse
import logging
import os
import pickle as pkl
from posixpath import join

np.set_printoptions(suppress=True)
cuda_device = 1
store_nmiMatrix_path = '../nmiMatrix'


def get_data(mode_name, matrix_store_path, proportion):
    label_with_timestamp, metric_dataset, log_dataset, trace_dataset = get_channels(args.service_s, mode_name, proportion)
    metric_shape = metric_dataset.shape[1]
    log_shape = log_dataset.shape[1]
    trace_shape = trace_dataset.shape[1]
    channels = pd.concat([metric_dataset, log_dataset, trace_dataset], axis=1)
    channels_columns = channels.columns
    channels = channels.reset_index(drop=True)
    channels = channels.values.T
    print(channels)
    channels = normalize(channels, axis=1, norm='max')
    
    if not os.path.exists(join(store_nmiMatrix_path, matrix_store_path)):
        nmiMatrix1 = np.array([[0 for col in range(channels.shape[0])]
                              for row in range(channels.shape[0])])  # 这部分计算比较耗时，可以存储在文件中
        nmiMatrix2 = np.array([[0 for col in range(channels.shape[0])]
                              for row in range(channels.shape[0])])
        nmiMatrix3 = np.array([[0 for col in range(channels.shape[0])]
                              for row in range(channels.shape[0])])
        nmiMatrix4 = np.array([[0 for col in range(channels.shape[0])]
                              for row in range(channels.shape[0])])
        nmiMatrix5 = np.array([[0 for col in range(channels.shape[0])]
                              for row in range(channels.shape[0])])
        nmiMatrix6 = np.array([[0 for col in range(channels.shape[0])]
                              for row in range(channels.shape[0])])

        for i1 in range(metric_shape):
            for j1 in range(metric_shape):
                nmiMatrix1[i1][j1] = mutual_info_score(channels[i1], channels[j1])
        for i2 in range(metric_shape, metric_shape + log_shape):
            for j2 in range(metric_shape, metric_shape + log_shape):
                nmiMatrix2[i2][j2] = mutual_info_score(
                    channels[i2], channels[j2])

        for i3 in range(metric_shape + log_shape, metric_shape+log_shape+trace_shape):
            for j3 in range(metric_shape + log_shape, metric_shape+log_shape+trace_shape):
                nmiMatrix3[i3][j3] = mutual_info_score(
                    channels[i3], channels[j3])

        # metric & log
        for i4 in range(metric_shape):
            for j4 in range(metric_shape, metric_shape + log_shape):
                nmiMatrix4[i4][j4] = mutual_info_score(
                    channels[i4], channels[j4])
        for i4 in range(metric_shape, metric_shape + log_shape):
            for j4 in range(metric_shape):
                nmiMatrix4[i4][j4] = mutual_info_score(
                    channels[i4], channels[j4])

        # metric & trace
        for i5 in range(metric_shape):
            for j5 in range(metric_shape + log_shape, metric_shape+log_shape+trace_shape):
                nmiMatrix5[i5][j5] = mutual_info_score(
                    channels[i5], channels[j5])
        for i5 in range(metric_shape + log_shape, metric_shape+log_shape+trace_shape):
            for j5 in range(metric_shape):
                nmiMatrix5[i5][j5] = mutual_info_score(
                    channels[i5], channels[j5])

        # log & trace
        for i6 in range(metric_shape, metric_shape + log_shape):
            for j6 in range(metric_shape + log_shape, metric_shape+log_shape+trace_shape):
                nmiMatrix6[i6][j6] = mutual_info_score(
                    channels[i6], channels[j6])
        for i6 in range(metric_shape + log_shape, metric_shape+log_shape+trace_shape):
            for j6 in range(metric_shape, metric_shape + log_shape):
                nmiMatrix6[i6][j6] = mutual_info_score(
                    channels[i6], channels[j6])

        nmiMatrix = np.array(
            [nmiMatrix1, nmiMatrix2, nmiMatrix3, nmiMatrix4, nmiMatrix5, nmiMatrix6])
        print("nmiMatrix.shape:", nmiMatrix.shape)

        with open(join(store_nmiMatrix_path, matrix_store_path), 'wb') as f:
            pkl.dump(nmiMatrix, f)

    else:
        with open(join(store_nmiMatrix_path, matrix_store_path), 'rb') as f:
            nmiMatrix = pkl.load(f)
    return channels_columns, label_with_timestamp, channels, nmiMatrix


def eval(label_with_timestamp, model_path, test_loader):
    with torch.no_grad():
        if model_path:
            net = torch.load(model_path).cuda(cuda_device).eval()
            distance_frame = pd.DataFrame(
                columns=['timetamp', 'distance', 'label'])
            for _, (batch_label, batch_aj, batch_channel, batch_timestamp) in enumerate(tqdm(test_loader)):
                X = batch_channel
                A = batch_aj
                X = X.float().cuda(cuda_device)
                A = A.float().cuda(cuda_device)
                batch_label = torch.tensor(batch_label).cuda(cuda_device)
                label = np.array(
                    batch_label.squeeze().cpu().numpy(), dtype=np.double)
                output = net(X, A)
                pred = np.array(output.cpu().numpy(), dtype=np.double)
                batch_timestamp = np.array(batch_timestamp.cpu().numpy())
                for t in range(batch_timestamp.shape[0]):
                    ground_truth = label_with_timestamp[label_with_timestamp['timestamp'] == batch_timestamp[t][0]]
                    ground_truth = ground_truth['label'].values
                    
                    ed_dis = []
                    err = []
                    for m in range(len(pred[t])):
                        err.append(abs(pred[t][m]-label[t][m]))
                    for m in range(len(pred[t])):
                        ed_dis_m = (abs(
                            pred[t][m]-label[t][m])-np.median(np.array(err)))/np.percentile(np.array(err), 25)
                        ed_dis.append(ed_dis_m)
                    distance_frame = distance_frame.append(pd.DataFrame({'timetamp': [batch_timestamp[t][0]],
                                                                         'distance': max(ed_dis),
                                                                         'label': ground_truth}))

    distance_frame = distance_frame.sort_values('timetamp')
    distance_frame.to_csv(args.service_s+'_ed.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='training of the feed-forward extractor (ff-ext, ML)'
    )
    parser.add_argument('--mode', required=True, help='run in running mode')
    parser.add_argument('--service_s', required=True, help='the service name')
    parser.add_argument('--version', type=int, action='store', help='the evaluate version')
    parser.add_argument('--epoch_num', type=int, default=100, help='the epoch of training')
    parser.add_argument('--batch_size', type=int, default=64, help='the batch_size of training')
    parser.add_argument('--window_size', type=int, default=20, help='the windowsize of data')  # 10分钟
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        filename='./train.log',
                        filemode='a',
                        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                        )

    checkpoint_path = '../checkpoint'
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
        
    if args.mode == 'train':
        print('====================train===========================')
        channels_columns, label_with_timestamp, channels, nmiMatrix = get_data(
            'train', args.service_s+'train_nmiMatrix.pk', 0.6)
        print("Now channels shape:", channels.shape)

        train_data = MyTorchDataset(label_with_timestamp=label_with_timestamp,
                                            channels=channels, aj_matrix=nmiMatrix, window_size=args.window_size)
        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
        dataset_length = int(len(train_data)/args.batch_size)

        # 定义损失函数和优化器
        node_num = channels.shape[0]  # 节点的个数
        edge_types = 6  # 边的种类
        net = Net(node_num=node_num, edge_types=edge_types, window_samples_num=args.window_size, dropout=0.1).cuda(cuda_device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-5, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1, verbose=True)

        model_version = 0
        for i in range(100):
            if not os.path.exists(join(checkpoint_path, f'model_save_v{i}')):
                model_version = i
                os.mkdir(join(checkpoint_path, f'model_save_v{i}'))
                model_path = join(
                    join(checkpoint_path, f'model_save_v{i}'))
                break

        for epoch in range(args.epoch_num):
            distance_frame = pd.DataFrame(
                columns=['timetamp', 'distance', 'label'])
            running_loss = 0.0
            count = 0
            with tqdm(total=dataset_length) as pbar:
                for step, (batch_label, batch_aj, batch_channel, batch_timestamp) in enumerate(train_loader):
                    count = step
                    X = batch_channel
                    A = batch_aj
                    X = X.float().cuda(cuda_device)
                    A = A.float().cuda(cuda_device)

                    # prepare for output show
                    t = batch_timestamp.squeeze().cpu().numpy()
                    output = net(X, A)
                    batch_label = torch.tensor(
                        batch_label).cuda(cuda_device)
                    label = np.array(
                        batch_label.squeeze().cpu().numpy(), dtype=np.double)

                    loss = criterion(
                        output, batch_label.float().squeeze())  # MSE Loss
                    loss = loss.sum()
                    loss.backward()
                    optimizer.step()
                    net.zero_grad()

                    running_loss += loss.item()
                    pred = np.array(
                        output.cpu().detach().numpy(), dtype=np.double)

                    pbar.set_postfix(loss=loss.item(),
                                        epoch=epoch, v_num=model_version)
                    pbar.update(1)

            count += 1
            logging.info(('epoch:', str(epoch), 'loss:', str(running_loss/count),))
            scheduler.step(running_loss/count)
            torch.save(net, model_path + '/checkpoint_'+args.service_s+'_' +
                        str(epoch) + '_' + str(running_loss/count)[:6]+'_model.pkl')

    if args.mode == 'eval':
        channels_columns, label_with_timestamp, test_channels, test_nmiMatrix = get_data(
            'test', args.service_s+'test_nmiMatrix.pk', 0.6)
        test_data = MyTorchDataset(label_with_timestamp=label_with_timestamp,
                                        channels=test_channels, aj_matrix=test_nmiMatrix, window_size=args.window_size)
        test_loader = DataLoader(
            dataset=test_data, batch_size=args.batch_size, shuffle=False, drop_last=True)

        eval(label_with_timestamp, '../checkpoint/model_save_v2/checkpoint_mobservice2_0_0.2996_model.pkl', test_loader)