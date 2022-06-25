import os
import ntpath
import time
from . import util
from . import html
import numpy as np
import scipy.misc
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x


class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.tf_log = opt.isTrain and opt.tf_log
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        web_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.img_dir = os.path.join(web_dir, 'images')
        print('create directory %s...' % self.img_dir)
        util.mkdirs([self.img_dir])
        if opt.isTrain:
            self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):

        ## convert tensors to numpy arrays
        visuals = self.convert_visuals_to_numpy(visuals)

        img_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.7d.png' % (epoch, step))
        visuals_lst = []
        for label, image_numpy in visuals.items():
            if len(image_numpy.shape) >= 4:
                image_numpy = image_numpy[0]
            visuals_lst.append(image_numpy)
        image_cath = np.concatenate(visuals_lst, axis=0)
        print("Saving", img_path)
        util.save_image(image_cath, img_path)

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        if self.tf_log:
            for tag, value in errors.items():
                value = value.mean().float()
                summary = self.tf.Summary(value=[self.tf.Summary.Value(tag=tag, simple_value=value)])
                self.writer.add_summary(summary, step)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            #print(v)
            #if v != 0:
            v = v.mean().float()
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def convert_visuals_to_numpy(self, visuals):
        for key, t in visuals.items():
            tile = self.opt.batchSize > 8
            if 'input_label' == key:
                t = util.tensor2label(t, self.opt.label_nc, tile=tile)
            else:
                t = util.tensor2im(t, tile=tile)
            visuals[key] = t
        return visuals

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):        
        visuals = self.convert_visuals_to_numpy(visuals)        

        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        visuals_lst = []
        image_name = '%s.png' % name
        save_path = os.path.join(image_dir, image_name)
        for label, image_numpy in visuals.items():
            if len(image_numpy.shape) >= 4:
                image_numpy = image_numpy[0]
            visuals_lst.append(image_numpy)

        image_cath = np.concatenate(visuals_lst, axis=0)
        util.save_image(image_cath, save_path, create_dir=True)
