import argparse
import os
from util import util
import torch
import models
import data


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self):
        """Define the common options that are used in both training and test."""
        defaults=dict()
        # basic parameters
        defaults['txt_file_A'] = True
        defaults['txt_file_B'] = True
        defaults['name'] = 'experiment_name'
        defaults['gpu_ids'] = '0'
        defaults['checkpoints_dir'] = './checkpoints'
        defaults['model'] = './cycle_gan'
        defaults['input_nc'] = 1
        defaults['output_nc'] = 1
        defaults['ngf'] = 64
        defaults['ndf'] = 64
        defaults['netD'] = 'basic'
        defaults['netG'] = 'resnet_9blocks'
        defaults['n_layers_D'] = 3
        defaults['norm'] = 'instance'
        defaults['init_type'] = 'normal'
        defaults['init_gain'] = 0.02
        defaults['no_dropout'] = True
        defaults['dataset_mode'] = 'hdf5'
        defaults['direction'] = 'AtoB'
        defaults['serial_batches'] = True
        defaults['num_threads'] = 4
        defaults['batch_size'] = 1
        defaults['load_size'] = 512
        defaults['crop_size'] = 512
        defaults['max_dataset_size'] = float("inf")
        defaults['preprocess'] = "none"
        defaults['no_flip'] = True
        defaults['display_winsize'] = 256
        defaults['epoch'] = "latest"
        defaults['load_iter'] = "0"
        defaults['verbose'] = True
        defaults['suffix'] = ''
        self.initialized = True
        return defaults

    def gather_options(self, opt):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            defaults = self.initialize()

        # modify model-related parser options
        model_name = opt["model"]
        model_option_setter = models.get_option_setter(model_name)
        opt, defaults = model_option_setter(opt, defaults, self.isTrain)

        # modify dataset-related parser options
        dataset_name = opt["dataset_mode"]
        dataset_option_setter = data.get_option_setter(dataset_name)
        opt, defaults = dataset_option_setter(opt, defaults, self.isTrain)

        # update default options with user provided options
        opt = dict(defaults, **opt)

        # save and return the options dictionary
        self.opt = opt
        return opt

    def print_options(self, opt):
        """Print and save options

        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(opt.items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt['name'])
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt['phase']))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def process_options(self, opt):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options(opt)
        opt['isTrain'] = self.isTrain   # train or test

        # process opt.suffix
        if opt['suffix']:
            suffix = ('_' + opt['suffix'].format(**opt)) if opt['suffix'] != '' else ''
            opt['name'] = opt['name'] + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt['gpu_ids'].split(',')
        opt['gpu_ids'] = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt['gpu_ids'] = opt['gpu_ids'].append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt['gpu_ids'][0])

        self.opt = opt
        return self.opt
