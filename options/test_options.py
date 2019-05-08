from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self):
        opt = super(TestOptions, self).initialize()
        opt['ntest']=float("inf")
        opt['results_dir'] = './results/'
        opt['aspect_ratio'] = 1.0
        opt['phase'] = 'test'
        opt['eval'] = True
        opt['num_test'] = 50
        if 'model' not in opt:
            opt['model'] = 'test'
        # To avoid cropping, the load_size should be the same as crop_size
        if 'load_size' not in opt:
            opt['load_size'] = opt['crop_size']
        self.isTrain = False
        return opt
