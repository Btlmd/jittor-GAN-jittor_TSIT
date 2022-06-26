import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from tqdm import tqdm

import jittor as jt

# Use CUDA
jt.flags.use_cuda = 2  # Force CUDA

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)

visualizer = Visualizer(opt)

img_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))

# test
print('Number of images: ', len(dataloader))
for i, data_i in enumerate(tqdm(dataloader)):
    if i * opt.batchSize >= opt.how_many:
        break

    generated = model(data_i, mode='inference')

    img_path = data_i['cpath']
    for b in range(generated.shape[0]):
        visuals = OrderedDict([('synthesized_image', generated[b])])
        visualizer.save_images(img_dir, visuals, img_path[b:b + 1])

