from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self):
        defaults = super(TrainOptions, self).initialize()
        # visdom and HTML visualization parameters
        defaults['display_freq'] = 400
        defaults['display_ncols'] = 4
        defaults['display_id'] = 1
        defaults['display_server'] = "http://localhost"
        defaults['display_env'] = "main"
        defaults['display_port'] = 8097
        defaults['update_html_freq'] = 1000
        defaults['print_freq'] = 100
        defaults['no_html'] = True
        # network saving and loading parameters
        defaults['save_latest_freq'] = 5000
        defaults['save_epoch_freq'] = 5
        defaults['save_by_iter'] = True
        #defaults['continue_train'] = True
        defaults['continue_train'] = False
        defaults['epoch_count'] = 1
        defaults['phase'] = 'train'
        # training parameters
        defaults['niter'] = 100
        defaults['niter_decay'] = 100
        defaults['beta1'] = 0.5
        defaults['lr'] = 0.0002
        defaults['gan_mode'] = 'lsgan'
        defaults['pool_size'] = 50
        defaults['lr_policy'] = 'linear'
        defaults['lr_decay_iters'] = 50
        self.isTrain = True
        return defaults
