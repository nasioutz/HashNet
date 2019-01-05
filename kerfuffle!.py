import train
import test
from os.path import join
import json


# ------ TRAIN ------

args_train = train.Arguments(net='AlexNet', hash_bit=48, dual_batch=False, scale_tanh=False, tanh_step=200,
                             #snapshot=("cross-entropy", '2018_12_25_4_19_4', 'iter_07500'),
                             pretrained=True, alternative_alexnet=False, alt_version=2,
                             workers=8, batch_size=128, resize_size=256, crop_size=224, num_iterations=2000,
                             lr=0.005, lr_step=200, lr_decay_factor=0.1,
                             loss_type="cauchy-cross-entropy", gamma=20, q_lambda=0.0, class_num=56)

config_train = train.produce_config(args_train)

#print()
#print(config_train["loss"])

config_train["out_file"].write(json.dumps(config_train["network"]["params"])+"\n")
config_train["out_file"].write(json.dumps(config_train["loss"])+"\n")
config_train["out_file"].write(json.dumps(config_train["optimizer"])+"\n")
config_train["out_file"].write(json.dumps(config_train["prep"])+"\n")
config_train["out_file"].write(json.dumps(config_train["num_iterations"])+"\n")
config_train["out_file"].write(json.dumps(config_train["snapshot_interval"])+"\n")
config_train["out_file"].write(json.dumps(config_train["hash_bit"])+"\n")

train.train(config_train)

# ------ TEST ------

args_test = test.Arguments(snapshot=(args_train.loss_type, args_train.prefix, 'iter_01500'),
                           portion_oo_10=5, batch_size=16, crop10=False,
                           resize_size=args_train.resize_size, crop_size=args_train.crop_size,
                           hash_bit=args_train.hash_bit
                           )

config_test = test.produce_config(args_test)

code_and_label = test.predict(config_test)

mAP = test.mean_average_precision(code_and_label, config_test["R"])
print(config_test["snapshot_path"])
print(args_test.snapshot[0] + ": " + args_test.snapshot[1] + " | MAP: " + str(mAP))
print("saving ...")
test.save_code_and_label(code_and_label, join(config_test["output_path"], args_test.snapshot[0]))
config_test["out_file"].write(args_test.snapshot[0] + ": " + args_test.snapshot[1] + " | MAP: " + str(mAP))
print("saving done")

