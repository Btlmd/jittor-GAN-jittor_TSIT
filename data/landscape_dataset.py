import glob
import os
from pathlib import Path

# import torch

from data.pix2pix_dataset import Pix2pixDataset
# from data.image_folder import make_dataset


class LandScapeDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='fixed')
        parser.set_defaults(load_size=512)
        parser.set_defaults(crop_size=512)
        parser.set_defaults(display_winsize=512)
        parser.set_defaults(label_nc=255)
        parser.set_defaults(aspect_ratio=(512 / 384))
        parser.set_defaults(batchSize=1)
        parser.set_defaults(no_instance=True)
        return parser

    def get_paths(self, opt):
        root = opt.croot
        phase = 'val' if opt.phase == 'test' else 'train'

        # all_images = make_dataset(root, recursive=True, read_cache=False, write_cache=False)
        # image_paths = []
        # label_paths = []
        # for p in all_images:
        #     if '_%s_' % phase not in p:
        #         continue
        #     if p.endswith('.jpg'):
        #         image_paths.append(p)
        #     elif p.endswith('.png'):
        #         label_paths.append(p)

        # These codes are not tested yet.
        # These pictures are needed to resize to 512 x 384 first !!!!!
        if opt.phase == 'train':
            image_paths = sorted(glob.glob(os.path.join(root, 'train', 'imgs', '*.jpg')))
            label_paths = sorted(glob.glob(os.path.join(root, 'train', 'labels', '*.png')))

        elif opt.phase == 'test':
            # image_paths = sorted(glob.glob(os.path.join(root, 'train', 'imgs', '*.jpg')))[0:100]
            # label_paths = sorted(glob.glob(os.path.join(root, 'train', 'labels', '*.png')))[0:100]
            # image_paths = sorted(glob.glob(os.path.join(root, 'test', 'imgs', '*.jpg')))
            label_paths = sorted(glob.glob(os.path.join(root, 'test', '*.png')))
            image_paths = label_paths

        print(root, len(image_paths), len(label_paths))

        instance_paths = []  # don't use instance map for ade20k

        return label_paths, image_paths, instance_paths


    # No postprocess at all
    def postprocess(self, input_dict):
        input_dict['label'] = (input_dict['label'] / 255.0).round()
        return input_dict