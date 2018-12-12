import train
import test
from os.path import join


# ------ TRAIN ------

args_train = train.Arguments(hash_bit=48, net='AlexNet', class_num=52.0, workers=0, batch_size=64)

config_train = train.produce_config(args_train)

print()
print(config_train["loss"])

train.train(config_train)

# ------ TEST ------

args_test = test.Arguments(snapshot=(args_train.prefix, 'iter_09000'), portion_oo_10=10)

config_test = test.produce_config(args_test)

code_and_label = test.predict(config_test)

mAP = test.mean_average_precision(code_and_label, config_test["R"])
print(config_test["snapshot_path"])
print("MAP: " + str(mAP))
print("saving ...")
test.save_code_and_label(code_and_label, join(config_test["output_path"], args_test.snapshot[0]))
print("saving done")

