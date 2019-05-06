import os
import random
import numpy as np
from PIL import Image
import h5py
from data.base_dataset import BaseDataset
import torchvision.transforms as transforms


IMG_EXTENSIONS = ['.dcm', '.DCM']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(txt_file, max_dataset_size=float("inf")):
    assert os.path.exists(txt_file), '%s is not a valid file' % txt_file

    fin = open(txt_file)
    images = [line.strip().split('\t')[0] for line in iter(fin.readline, '')]

    return images[:min(max_dataset_size, len(images))]


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True


def get_transform(opt, params=None, mode='I;16', grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = [transforms.ToPILImage(mode=mode)]

    # unknown behavior for images that range from 0 to 65535
    # if grayscale:
    # transform_list.append(transforms.Grayscale(1))

    # pre-process methods
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method)))
    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))
    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((32767.5,), (32767.5,))]
        else:
            transform_list += [transforms.Normalize((32767.5, 32767.5, 32767.5), (32767.5, 32767.5, 32767.5))]
        """
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        """
        """
        if opt.minmax:
            transforms.append(Lambda(lambda x: ((x - torch.min(x)) / (torch.max(x) - torch.min(x)))))
            print("minmax applied, {}".format(minmax))
        scale_1_1 = self.train_kwargs.get('scale_1_1', False)
        if scale_1_1:
            transforms.append(Lambda(lambda x: ((x - torch.min(x)) / (torch.max(x) - torch.min(x))) * 2.0 - 1.0))
            print("scale_1_1 applied, {}".format(scale_1_1))
        """
    return transforms.Compose(transform_list)


class Hdf5Dataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired hdf5 datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.txt_file_A = opt.txt_file_A
        self.txt_file_B = opt.txt_file_B

        self.A_paths = sorted(make_dataset(self.txt_file_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.txt_file_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        # Read images
        A_img = self.read_img(A_path)
        B_img = self.read_img(B_path)

        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

    def read_img(self, img_path):
        """
        Read dicoms
        """
        image, _ = read_hdf5(img_path)

        # Gray scale add channel axis
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]

        return image


def read_hdf5(filename, read_keys=["data", "label"]):
    with h5py.File(filename, 'r') as fin:
        data_dict = {}
        for k in read_keys:
            if k == "data":
                data_dict["data"] = np.squeeze(np.array(fin[k])).astype(np.uint16)
            if k == "label":
                data_dict["label"] = np.squeeze(np.array(fin[k])).astype(np.int64)
    return data_dict["data"], data_dict["label"]

