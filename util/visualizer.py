import os
import time
import ntpath
from collections import OrderedDict
import wandb
from . import util, html

def save_images(visuals, image_path, web_dir, aspect_ratio=1.0, width=256, use_wandb=False):
    """Save images and log them to wandb.

    Parameters:
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        web_dir (str)            -- the base directory where images will be saved
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' and log them using wandb.
    """
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    ims_dict = {}
    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        label_dir = os.path.join(web_dir, label)  # Separate folder for each label
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        save_path = os.path.join(label_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        if use_wandb:
            ims_dict[label] = wandb.Image(im, caption=label)
    if use_wandb:
        wandb.log(ims_dict)
    

class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information using wandb."""

    def __init__(self, opt):
        """Initialize the Visualizer class
        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt  # cache the option
        self.use_wandb = opt.use_wandb
        self.wandb_project_name = opt.wandb_project_name
        self.name = opt.name
        self.saved = False
        self.current_epoch = 0

        if self.use_wandb:
            self.wandb_run = wandb.init(project=self.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
            self.wandb_run._label(repo='CycleGAN-and-pix2pix')

        if opt.isTrain and not opt.no_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])

        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def display_current_results(self, visuals, epoch, save_result):
        """Display current results on wandb; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        if self.use_wandb:
            ims_dict = {}
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                ims_dict[label] = wandb.Image(image_numpy)
            self.wandb_run.log(ims_dict)

        if save_result or not self.saved:  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []
                for label, image_numpy in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=256)  # width can be adjusted
            webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """Display the current losses on wandb

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if self.use_wandb:
            self.wandb_run.log(losses)

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """Print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
            