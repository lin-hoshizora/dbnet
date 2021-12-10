import json
import numpy as np
from dbnet.dataset.tfrecord import gen_tfr, split


# with open("/datasets/almex_insurance/label.json") as f:
#     labels = json.load(f)["_via_img_metadata"]
with open("./save_500_int.json") as f:
    labels = json.load(f)["_via_img_metadata"]    
    
np.random.seed(42)
ds = list(labels.values())
np.random.shuffle(ds)




# gen_tfr(ds=ds, folder="/datasets/e2e/train/", save_path="/datasets/almex_insurance/almex.tfrecord")
gen_tfr(ds=ds, folder="/datasets/almex_insurance/img/", save_path="./lin/train_500_p.tfrecord")
split(path="./lin/train_500.tfrecord", n_val=50)
