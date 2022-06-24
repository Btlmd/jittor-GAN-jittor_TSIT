import torch
import pickle
import hashlib
from pathlib import Path
from IPython import embed

fd = Path("checkpoints/sis_landscape")


def load_jtpkl(path):
    import jittor
    sd = jittor.load(path)
    embed()


def safepickle(obj, path):
    # Protocol version 4 was added in Python 3.4. It adds support for very large objects, pickling more kinds of objects, and some data format optimizations.
    # ref: <https://docs.python.org/3/library/pickle.html>
    s = pickle.dumps(obj, 4)
    checksum = hashlib.sha1(s).digest()
    s += bytes(checksum)
    s += b"HCAJSLHD"
    with open(path, 'wb') as f:
        f.write(s)

# load_jtpkl("/work/lambda/jt_tsit/checkpoints/sis_landscape/initial_net_G.pkl")
# exit(0)

tag = ".pth"
for f in fd.iterdir():
    if f.is_file() and f.name[-len(tag):] == tag:
        print("Converting", f.name)
        sd = torch.load(f)
        new_sd = {}
        # embed()
        for k, v in sd.items():
            if "num_batches_tracked" in k:
                continue

            if "weight_orig" in k:
                new_sd[k] = v.numpy()
                new_sd[k.replace("_orig", "")] = v.numpy()
                continue

            new_sd[k] = v.numpy()
        print("Saving", (f.name[:-len(tag)] + ".pickle"))
        with open(fd / (f.name[:-len(tag)] + ".pickle"), "wb") as f:
            pickle.dump(new_sd, f)

# import jittor as jt
# tag = ".pickle"
# for f in fd.iterdir():
#     if f.is_file() and f.name[-len(tag):] == tag:
#         print("Converting", f.name)
#         with open(f, "rb") as input_file:
#             sd = pickle.load(input_file)
#         print("Saving", (f.name[:-len(tag)] + ".pkl"))
#         jt.save(sd, fd / (f.name[:-len(tag)] + ".pkl"))
