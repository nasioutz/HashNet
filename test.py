import argparse
import os
import os.path as osp
from torch import nn
import numpy as np
import torch
import pre_process as prep
import torch.utils.data as util_data
from data_list import ImageList
from torch.autograd import Variable
from os.path import join
from tqdm import tqdm


def save_code_and_label(params, path):
    database_code = params['database_code']
    validation_code = params['test_code']
    database_labels = params['database_labels']
    validation_labels = params['test_labels']
    np.save(path + "_database_code.npy", database_code)
    np.save(path + "_database_labels.npy", database_labels)
    np.save(path + "_test_code.npy", validation_code)
    np.save(path + "_test_labels.npy", validation_labels)


def mean_average_precision(params, R):
    database_code = params['database_code']
    validation_code = params['test_code']
    database_labels = params['database_labels']
    validation_labels = params['test_labels']
    query_num = validation_code.shape[0]

    # sim = np.dot(database_code, validation_code.T)
    # ids = np.argsort(-sim, axis=0)

    ids = np.argsort(-np.dot(database_code, validation_code.T), axis=0)

    APx = []

    for i in tqdm(range(query_num)):
        label = validation_labels[i, :]
        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(database_labels[idx[0:R], :] == label, axis=1) > 0
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R + 1, 1)
        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)

    return np.mean(np.array(APx))


def code_predict(loader, model, name, test_10crop=True, gpu=True, poo10=10):
    start_test = True
    if test_10crop:
        iter_test = [iter(loader[name + str(i)]) for i in range(10)]
        for i in tqdm(range(len(loader[name + '0']))):
            data = [iter_test[j].next() for j in range(10)]
            inputs = [data[j][0] for j in range(10)]
            labels = data[0][1]
            if gpu:
                for j in range(10):
                    inputs[j] = Variable(inputs[j].cuda())
            else:
                for j in range(10):
                    inputs[j] = Variable(inputs[j])
            outputs = []
            for j in range(10):
                outputs.append(model(inputs[j]))
            outputs = sum(outputs) / 10.0
            if start_test:
                all_output = outputs.data.float()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    else:
        iter_val = iter(loader[name])
        for i in tqdm(range(len(loader[name]) // 10 * poo10)):
            data = iter_val.next()
            inputs = data[0]
            labels = data[1]
            if gpu:
                inputs = Variable(inputs.cuda())
            else:
                inputs = Variable(inputs)
            outputs = model(inputs)
            if start_test:
                all_output = outputs.data.cpu().float()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.cpu().float()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    return torch.sign(all_output), all_label


def predict(config):
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    if prep_config["test_10crop"]:
        prep_dict["database"] = prep.image_test_10crop( \
            resize_size=prep_config["resize_size"], \
            crop_size=prep_config["crop_size"])
        prep_dict["test"] = prep.image_test_10crop( \
            resize_size=prep_config["resize_size"], \
            crop_size=prep_config["crop_size"])
    else:
        prep_dict["database"] = prep.image_test( \
            resize_size=prep_config["resize_size"], \
            crop_size=prep_config["crop_size"])
        prep_dict["test"] = prep.image_test( \
            resize_size=prep_config["resize_size"], \
            crop_size=prep_config["crop_size"])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    if prep_config["test_10crop"]:
        for i in tqdm(range(10)):
            dsets["database" + str(i)] = ImageList(open(data_config["database"]["list_path"]).readlines(), \
                                                   transform=prep_dict["database"]["val" + str(i)])
            dset_loaders["database" + str(i)] = util_data.DataLoader(dsets["database" + str(i)], \
                                                                     batch_size=data_config["database"]["batch_size"], \
                                                                     shuffle=False, num_workers=8)
            dsets["test" + str(i)] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                               transform=prep_dict["test"]["val" + str(i)])
            dset_loaders["test" + str(i)] = util_data.DataLoader(dsets["test" + str(i)], \
                                                                 batch_size=data_config["test"]["batch_size"], \
                                                                 shuffle=False, num_workers=8)

    else:
        dsets["database"] = ImageList(open(data_config["database"]["list_path"]).readlines(), \
                                      transform=prep_dict["database"])
        dset_loaders["database"] = util_data.DataLoader(dsets["database"], \
                                                        batch_size=data_config["database"]["batch_size"], \
                                                        shuffle=False, num_workers=8)
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                  transform=prep_dict["test"])
        dset_loaders["test"] = util_data.DataLoader(dsets["test"], \
                                                    batch_size=data_config["test"]["batch_size"], \
                                                    shuffle=False, num_workers=8)
    ## set base network
    base_network = nn.Sequential(torch.load(config["snapshot_path"]))

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        base_network = base_network.cuda()

    database_codes, database_labels = code_predict(dset_loaders, base_network, "database",
                                                   test_10crop=prep_config["test_10crop"], gpu=use_gpu,
                                                   poo10=config['portion_oo_10'])
    test_codes, test_labels = code_predict(dset_loaders, base_network, "test", test_10crop=prep_config["test_10crop"],
                                           gpu=use_gpu, poo10=config['portion_oo_10'])

    return {"database_code": database_codes.numpy(), "database_labels": database_labels.numpy(), \
            "test_code": test_codes.numpy(), "test_labels": test_labels.numpy()}


class Arguments:
    def __init__(self, snapshot, gpu_id='0', dataset='nus_wide', hash_bit=48, portion_oo_10=10, batch_size=16,
                 crop_size=224,resize_size=256,crop10=True):
        from time import localtime

        timestamp = str(localtime().tm_year) + '_' + str(localtime().tm_mon) + '_' + str(
            localtime().tm_mday) + '_' + str(
            localtime().tm_hour) + '_' + str(localtime().tm_min) + '_' + str(localtime().tm_sec)

        self.portion_oo_10 = portion_oo_10
        self.snapshot = snapshot
        self.prefix = timestamp
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        self.dataset = dataset
        self.hash_bit = hash_bit
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.crop10 = crop10


def produce_config(args,old_snapshot=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # train config
    config = {}
    config["portion_oo_10"] = args.portion_oo_10
    config["dataset"] = args.dataset


    config["prep"] = {"test_10crop": args.crop10, "resize_size": args.resize_size, "crop_size": args.crop_size}
    if config["dataset"] == "imagenet":
        config["data"] = {
            "database": {"list_path": join('data', 'imagenet', 'database.txt'), 'batch_size': args.batch_size}, \
            "test": {"list_path": join('data', 'imagenet', 'test.txt'), "batch_size": args.batch_size}}
        config["R"] = 1000
    elif config["dataset"] == "nus_wide":
        config["data"] = {
            "database": {"list_path": join('data', 'nuswide_81', 'database.txt'), 'batch_size': args.batch_size}, \
            "test": {"list_path": join('data', 'nuswide_81', 'test.txt'), 'batch_size': args.batch_size}}
        config["R"] = 5000
    elif config["dataset"] == "coco":
        config["data"] = {
            "database": {"list_path": join('data', 'coco', 'database.txt'), 'batch_size': args.batch_size}, \
            "test": {"list_path": join('data', 'coco', 'test.txt'), "batch_size": args.batch_size}}
        config["R"] = 5000

    if old_snapshot:
        config["snapshot_path"] = join("snapshot", config["dataset"], str(args.hash_bit),config["dataset"] + "_" + str(args.hash_bit) + "bit_" + args.snapshot[0],
                                   args.snapshot[1] + "_model.pth.tar")
    else:
        config["snapshot_path"] = join("snapshot", args.snapshot[0], config["dataset"], str(args.hash_bit),
                                       config["dataset"] + "_" + str(args.hash_bit) + "bit_" + args.snapshot[1],
                                       args.snapshot[2] + "_model.pth.tar")

    config["output_path"] = join('snapshot',args.snapshot[0], config["dataset"], str(args.hash_bit), config["dataset"] + "_" + str(args.hash_bit) + "bit_" + args.prefix+" [TEST]")
    if not osp.exists(config["output_path"]):
        os.makedirs(config["output_path"])
    config["out_file"] = open(join(config["output_path"], "log.txt"), "w")

    if not osp.exists(config["output_path"]):
        os.makedirs(config["output_path"])

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--dataset', type=str, default='coco', help="dataset name")
    parser.add_argument('--hash_bit', type=int, default=48, help="number of hash code bits")
    parser.add_argument('--prefix', type=str, help="save path prefix")
    parser.add_argument('--snapshot', type=str, help="model path prefix")
    args = parser.parse_args()

    args = Arguments(snapshot=("cauchy-cross-entropy",'2018_12_26_15_0_51', 'iter_01500'),
                     portion_oo_10=10, hash_bit=48, batch_size=16,
                     crop_size=256, resize_size=256, crop10=False)

    config = produce_config(args)

    code_and_label = predict(config)

    mAP = mean_average_precision(code_and_label, config["R"])
    print(config["snapshot_path"])
    print(args.snapshot[0]+": "+args.snapshot[1]+" | MAP: " + str(mAP))
    print("saving ...")
    save_code_and_label(code_and_label, osp.join(config["output_path"], args.snapshot[0]))
    config["out_file"].write(args.snapshot[0] + ": " + args.snapshot[1] + " | MAP: " + str(mAP))
    print("saving done")



