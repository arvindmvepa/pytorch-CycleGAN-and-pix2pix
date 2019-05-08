from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self):
        defaults = super(TestOptions, self).initialize()
        defaults['ntest']=float("inf")
        defaults['results_dir'] = './results/'
        defaults['aspect_ratio'] = 1.0
        defaults['phase'] = 'test'
        defaults['eval'] = True
        defaults['num_test'] = 50
        if 'model' not in defaults:
            defaults['model'] = 'test'
        # To avoid cropping, the load_size should be the same as crop_size
        if 'load_size' not in defaults:
            defaults['load_size'] = defaults['crop_size']
        self.isTrain = False
        return defaults
