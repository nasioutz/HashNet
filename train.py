import json
import argparse
import os
import os.path as osp
from os.path import join
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import pre_process as prep
import torch.utils.data as util_data
import lr_schedule
import data_list
from data_list import ImageList
from torch.autograd import Variable
from time import localtime
from tqdm import tqdm
from tqdm import trange

import test

optim_dict = {"SGD": optim.SGD, "Adam": optim.Adam}
loss_dict = {"cross-entropy": loss.pairwise_loss, "cauchy-cross-entropy": loss.cauchy_cross_entropy_loss}


def image_classification_predict(loader, model, test_10crop=True, gpu=True, softmax_param=1.0):
    start_test = True
    if test_10crop:
        iter_test = [iter(loader['test' + str(i)]) for i in range(10)]
        for i in range(len(loader['test0'])):
            data = [iter_test[j].next() for j in range(10)]
            inputs = [data[j][0] for j in range(10)]
            labels = data[0][1]
            if gpu:
                for j in range(10):
                    inputs[j] = Variable(inputs[j].cuda())
                labels = Variable(labels.cuda())
            else:
                for j in range(10):
                    inputs[j] = Variable(inputs[j])
                labels = Variable(labels)
            outputs = []
            for j in range(9):
                _, predict_out = model(inputs[j])
                outputs.append(nn.Softmax()(softmax_param * predict_out))
            outputs_center = model(inputs[9])
            outputs.append(nn.Softmax()(softmax_param * outputs_center))
            softmax_outputs = sum(outputs)
            outputs = outputs_center
            if start_test:
                all_output = outputs.data.float()
                all_softmax_output = softmax_outputs.data.cpu().float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_softmax_output = torch.cat((all_softmax_output, softmax_outputs.data.cpu().float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)
    else:
        iter_val = iter(loader["test"])
        for i in range(len(loader['test'])):
            data = iter_val.next()
            inputs = data[0]
            if gpu:
                inputs = Variable(inputs.cuda())
            else:
                inputs = Variable(inputs)
            _, outputs = model(inputs)
            softmax_outputs = nn.Softmax()(softmax_param * outputs)
            if start_test:
                all_output = outputs.data.cpu().float()
                all_softmax_output = softmax_outputs.data.cpu().float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.cpu().float()), 0)
                all_softmax_output = torch.cat((all_softmax_output, softmax_outputs.data.cpu().float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)
    _, predict = torch.max(all_output, 1)
    return all_softmax_output, predict, all_output, all_label


def image_classification_test(loader, model, test_10crop=True, gpu=True):
    start_test = True
    if test_10crop:
        iter_test = [iter(loader['test' + str(i)]) for i in range(10)]
        for i in range(len(loader['test0'])):
            data = [iter_test[j].next() for j in range(10)]
            inputs = [data[j][0] for j in range(10)]
            labels = data[0][1]
            if gpu:
                for j in range(10):
                    inputs[j] = Variable(inputs[j].cuda())
                labels = Variable(labels.cuda())
            else:
                for j in range(10):
                    inputs[j] = Variable(inputs[j])
                labels = Variable(labels)
            outputs = []
            for j in range(10):
                _, predict_out = model(inputs[j])
                outputs.append(nn.Softmax()(predict_out))
            outputs = sum(outputs)
            if start_test:
                all_output = outputs.data.float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)
    else:
        iter_test = iter(loader["test"])
        for i in range(len(loader['test'])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            if gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels)
            _, outputs = model(inputs)
            if start_test:
                all_output = outputs.data.float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label) / float(all_label.size()[0])
    return accuracy


def train(config):
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["train_set1"] = prep.image_train( \
        resize_size=prep_config["resize_size"], \
        crop_size=prep_config["crop_size"])
    prep_dict["train_set2"] = prep.image_train( \
        resize_size=prep_config["resize_size"], \
        crop_size=prep_config["crop_size"])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    dsets["train_set1"] = ImageList(open(data_config["train_set1"]["list_path"]).readlines(),
                                    transform=prep_dict["train_set1"])
    dset_loaders["train_set1"] = util_data.DataLoader(dsets["train_set1"],
                                                      batch_size=data_config["train_set1"]["batch_size"],
                                                      shuffle=True, num_workers=data_config["train_set2"]['workers'],
                                                      pin_memory=True)
    dsets["train_set2"] = ImageList(open(data_config["train_set2"]["list_path"]).readlines(),
                                    transform=prep_dict["train_set2"])
    dset_loaders["train_set2"] = util_data.DataLoader(dsets["train_set2"],
                                                      batch_size=data_config["train_set2"]["batch_size"],
                                                      shuffle=True, num_workers=data_config["train_set2"]['workers'],
                                                      pin_memory=True)

    hash_bit = config["hash_bit"]

    ## set base network
    net_config = config["network"]
    if net_config["net_snapshot"] is None:
        base_network = net_config["type"](**net_config["params"])
    else:
        base_network = torch.load(config["snapshot_path"])

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        base_network = base_network.cuda()

    ## collect parameters
    parameter_list = [{"params": base_network.feature_layers.parameters(), "lr": 1}, \
                      {"params": base_network.hash_layer.weight, "lr": 10},
                      {"params": base_network.hash_layer.bias, "lr": 20}]

    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optim_dict[optimizer_config["type"]](parameter_list, **(optimizer_config["optim_params"]))

    ## set loss function
    loss_config = config["loss"]
    loss_f = loss_dict[loss_config["type"]]
    param_lr = []

    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    ## train
    len_train1 = len(dset_loaders["train_set1"]) - 1
    len_train2 = len(dset_loaders["train_set2"]) - 1
    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    best_acc = 0.0
    t_range = trange(config["num_iterations"], desc="Loss Value", leave=True)
    for i in t_range:
        if i % config["snapshot_interval"] == 0 or i == config["num_iterations"]-1:
            if i == config["num_iterations"]-1: j = i+1
            else: j = i
            torch.save(base_network, osp.join(config["output_path"], \
                                                             "iter_{:05d}_model.pth.tar".format(j)))

        ## train one iter
        base_network.train(True)
        optimizer = lr_scheduler(param_lr, optimizer, i, **schedule_param)
        optimizer.zero_grad()

        if i % len_train1 == 0:
            iter1 = iter(dset_loaders["train_set1"])
        inputs1, labels1 = iter1.next()
        if use_gpu:
            inputs1, labels1, = Variable(inputs1).cuda(), Variable(labels1).cuda()
        else:
            inputs1, labels1 = Variable(inputs1), Variable(labels1)

        if config["dual_batch"]:
            if i % len_train2 == 0:
                iter2 = iter(dset_loaders["train_set2"])
            inputs2, labels2 = iter2.next()
            if use_gpu:
                inputs2, labels2 = Variable(inputs2).cuda(), Variable(labels2).cuda()
            else:
                inputs2, labels2 = Variable(inputs2), Variable(labels2)
        else:
            inputs2 = inputs1
            labels2 = labels1

        inputs = torch.cat((inputs1, inputs2), dim=0)

        outputs = base_network(inputs)

        similarity_loss = loss_f(outputs.narrow(0, 0, inputs1.size(0)), labels1,
                                 outputs.narrow(0, inputs1.size(0), inputs2.size(0)), labels2,
                                 config_loss=loss_config["loss_param"])

        similarity_loss.backward()

        if i > 0:
            t_range.set_description("Loss Value: %f" % similarity_loss.float())
        t_range.refresh()
        # print("Iter: {:05d}, loss: {:.3f}".format(i, similarity_loss.float().data[0]))

        config["out_file"].write("Iter: {:05d}, loss: {:.3f}".format(i, \
                                                                     similarity_loss.float()))
        optimizer.step()


class Arguments:
    def __init__(self, gpu_id='0', dataset='nus_wide', hash_bit=48, lr=0.0003, class_num=1.0, gamma=1.0, q_lambda=0.0,
                 net='ResNet50', alternative_alexnet=False, batch_size=36, workers=8, scale_tanh=True, num_iterations=10000,
                 lr_step=2000, loss_type="cross-entropy",crop10=True,pretrained=True,crop_size=224,resize_size=256, dual_batch=True,
                 tanh_step=200, snapshot=None, lr_decay_factor=0.5, alt_version=1):
        from time import localtime

        timestamp = str(localtime().tm_year) + '_' + str(localtime().tm_mon) + '_' + str(
            localtime().tm_mday) + '_' + str(
            localtime().tm_hour) + '_' + str(localtime().tm_min) + '_' + str(localtime().tm_sec)

        self.prefix = timestamp
        self.gpu_id = gpu_id
        self.dataset = dataset
        self.hash_bit = hash_bit
        self.lr = lr
        self.class_num = class_num
        self.net = net
        self.workers = workers
        self.batch_size = batch_size
        self.gamma = gamma
        self.q_lambda = q_lambda
        self.scale_tanh = scale_tanh
        self.lr_step = lr_step
        self.num_iterations = num_iterations
        self.loss_type = loss_type
        self.alternative_alexnet = alternative_alexnet
        self.crop10 = crop10
        self.pretrained = pretrained
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.tanh_step = tanh_step
        self.dual_batch = dual_batch
        self.snapshot = snapshot
        self.lr_decay_factor = lr_decay_factor
        self.alt_version = alt_version


def produce_config(args, old_snapshot=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # train config
    config = {}
    config["num_iterations"] = args.num_iterations
    config["snapshot_interval"] = 500
    config["dataset"] = args.dataset
    config["hash_bit"] = args.hash_bit
    config["dual_batch"] = args.dual_batch
    config["network"] = {}
    if "ResNet" in args.net:
        config["network"]["type"] = network.ResNetFc
        config["network"]["params"] = {"name": args.net
            , "hash_bit": config["hash_bit"]}
    elif "VGG" in args.net:
        config["network"]["type"] = network.VGGFc
        config["network"]["params"] = {"name": args.net, "hash_bit": config["hash_bit"]}
    elif "AlexNet" in args.net:
        config["network"]["type"] = network.AlexNetFc
        config["network"]["params"] = {"hash_bit": config["hash_bit"], "pretrained": args.pretrained,
                                       "scale_tanh": args.scale_tanh,"tanh_step": args.tanh_step,
                                       "alternative_model": args.alternative_alexnet, "alt_version": args.alt_version}
    config["network"]["net_snapshot"] = args.snapshot
    config["prep"] = {"test_10crop": args.crop10, "resize_size": args.resize_size, "crop_size": args.crop_size}
    config["optimizer"] = {"type": "SGD", "optim_params": {"lr": 1.0, "momentum": 0.9, \
                                                           "weight_decay": 0.0005, "nesterov": True}, "lr_type": "step", \
                           "lr_param": {"lr_decay_factor": args.lr_decay_factor, "step": args.lr_step, "init_lr": args.lr}}

    config["loss"] = {"type": args.loss_type,
                      "loss_param": {"l_weight": 1.0, "q_weight": 0, "l_threshold": 15.0, "sigmoid_param": 10. / config["hash_bit"],
                      "class_num": args.class_num, "gamma": args.gamma, "q_lambda": args.q_lambda, "normed": True}}

    if config["dataset"] == "imagenet":
        config["data"] = {
            "train_set1": {"list_path": join('data', 'imagenet', 'train.txt'), "batch_size": args.batch_size,
                           "workers": args.workers}, \
            "train_set2": {"list_path": join('data', 'imagenet', 'train.txt'), "batch_size": args.batch_size,
                           "workers": args.workers}}
    elif config["dataset"] == "nus_wide":
        config["data"] = {
            "train_set1": {"list_path": join('data', 'nuswide_81', 'train.txt'), "batch_size": args.batch_size,
                           "workers": args.workers}, \
            "train_set2": {"list_path": join('data', 'nuswide_81', 'train.txt'), "batch_size": args.batch_size,
                           "workers": args.workers}}
    elif config["dataset"] == "coco":
        config["data"] = {"train_set1": {"list_path": join('data', 'coco', 'train.txt'), "batch_size": args.batch_size,
                                         "workers": args.workers}, \
                          "train_set2": {"list_path": join('data', 'coco', 'train.txt'), "batch_size": args.batch_size,
                                         "workers": args.workers}}

    if args.snapshot is not None:
        if old_snapshot:
            config["snapshot_path"] = join("snapshot", config["dataset"], str(args.hash_bit),config["dataset"] + "_" + str(args.hash_bit) + "bit_" + args.snapshot[0],
                                       args.snapshot[1] + "_model.pth.tar")
        else:
            config["snapshot_path"] = join("snapshot", args.snapshot[0], config["dataset"], str(args.hash_bit),
                                           config["dataset"] + "_" + str(args.hash_bit) + "bit_" + args.snapshot[1],
                                           args.snapshot[2] + "_model.pth.tar")
    else:
        config["snapshot_path"] = None

    config["output_path"] = join('snapshot', config["loss"]["type"], config["dataset"], str(config["hash_bit"]),
                                 config["dataset"] + "_" + str(config["hash_bit"]) + "bit_" + args.prefix)

    if not osp.exists(config["output_path"]):
        os.makedirs(config["output_path"])
    config["out_file"] = open(join(config["output_path"], "log.txt"), "w")

    if not osp.exists(config["output_path"]):
        os.makedirs(config["output_path"])

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HashNet')
    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--dataset', type=str, default='coco', help="dataset name")
    parser.add_argument('--hash_bit', type=int, default=48, help="number of hash code bits")
    parser.add_argument('--net', type=str, default='ResNet50', help="base network type")
    parser.add_argument('--prefix', type=str, help="save path prefix")
    parser.add_argument('--lr', type=float, default=0.0003, help="learning rate")
    parser.add_argument('--class_num', type=float, default=2.0, help="positive negative pairs balance weight")
    args = parser.parse_args()

    args = Arguments(hash_bit=48, net='AlexNet', class_num=1000.0)

    config = produce_config(args)

    print()
    print(config["loss"])
    print()

    config["out_file"].write(json.dumps(config["loss"]))
    config["out_file"].write(json.dumps(config["dataset"]))
    config["out_file"].write(json.dumps(config["num_iterations"]))
    config["out_file"].write(json.dumps(config["snapshot_interval"]))
    config["out_file"].write(json.dumps(config["dataset"]))
    config["out_file"].write(json.dumps(config["hash_bit"]))

    train(config)
