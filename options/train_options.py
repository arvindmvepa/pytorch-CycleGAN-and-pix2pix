from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self):
        opt = super(TrainOptions, self).initialize()
        # visdom and HTML visualization parameters
        opt['display_freq'] = 400
        opt['display_ncols'] = 4
        opt['display_id'] = 1
        opt['display_server'] = "http://localhost"
        opt['display_env'] = "main"
        opt['display_port'] = 8097
        opt['update_html_freq'] = 1000
        opt['print_freq'] = 100
        opt['no_html'] = True
        # network saving and loading parameters
        opt['save_latest_freq'] = 5000
        opt['save_epoch_freq'] = 5
        opt['save_by_iter'] = True
        opt['continue_train'] = True
        opt['epoch_count'] = 1
        opt['phase'] = 'train'
        # training parameters
        opt['niter'] = 100
        opt['niter_decay'] = 100
        opt['beta1'] = 0.5
        opt['lr'] = 0.0002
        opt['gan_mode'] = 'lsgan'
        opt['pool_size'] = 50
        opt['lr_policy'] = 'linear'
        opt['lr_decay_iters'] = 50
        self.isTrain = True
        return opt
