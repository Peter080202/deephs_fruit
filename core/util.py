import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for terminal/cluster
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import animation
from sklearn.cluster import KMeans
from PIL import Image
import gc
import torch
import os

from core.name_convention import *
import core.spectral_io as spectral_io

# Default folder to save plots
SAVE_DIR = "plots"
os.makedirs(SAVE_DIR, exist_ok=True)

# --------------------- Utility Functions ---------------------

def mask_background(_envi_data):
    _background_mask, _envi_data = get_background_mask(_envi_data)
    _background_mask = np.matmul(
        _background_mask, [[[True for i in range(_envi_data.shape[2])]]])
    _envi_data = np.ma.masked_array(_envi_data, mask=_background_mask)
    return _envi_data


def get_background_mask(_envi_data):
    avg_data = np.mean(_envi_data, 2, keepdims=True)
    _background_mask = avg_data < 0.2
    # replace border effects
    _background_mask[:, :150, :] = True
    _background_mask[:, 1970:, :] = True
    return _background_mask, _envi_data


def get_n_spectra(_envi_data, _num, _only_obj):
    if not _only_obj:
        _used_data = _envi_data.reshape(-1, _envi_data.shape[2])
        _pixel_ids = np.random.randint(0, _used_data.shape[0], _num)
        _spectra = _used_data[_pixel_ids, :]
    else:
        _masked_envi = mask_background(_envi_data)
        _used_data = _masked_envi.reshape(-1, _masked_envi.shape[2])
        _used_data = _used_data[np.logical_not(_used_data[:, 0].mask).squeeze()]
        _pixel_ids = np.random.randint(0, len(_used_data), _num)
        _spectra = _used_data[_pixel_ids, :]
    return _spectra


# --------------------- Plotting Functions ---------------------

def display_hyper_spectral_data(_data, _band=None, filename=None):
    fig = plt.figure()
    cmap = 'gray'

    if _band is None and _data.shape[2] > 3:
        ims = []
        for i in range(_data.shape[2]):
            ims.append([plt.imshow(_data[:, :, i].squeeze(), animated=True, cmap=cmap)])
        anim = animation.ArtistAnimation(fig, ims, interval=60, blit=True)
        if filename is None:
            filename = os.path.join(SAVE_DIR, "hyper_spectral_animation.gif")
        anim.save(filename, writer='imagemagick')
        print(f"# Saved hyperspectral animation: {filename}")
    elif _data.shape[2] == 3:
        plt.imshow(_data[:, :, :])
        if filename is None:
            filename = os.path.join(SAVE_DIR, "hyper_spectral_rgb.png")
        plt.savefig(filename)
        plt.close(fig)
        print(f"# Saved RGB image: {filename}")
    else:
        idx = 0 if _band is None else _band
        plt.imshow(_data[:, :, idx].squeeze(), cmap=cmap)
        if filename is None:
            filename = os.path.join(SAVE_DIR, f"hyper_spectral_band_{idx}.png")
        plt.savefig(filename)
        plt.close(fig)
        print(f"# Saved band image: {filename}")


def display_all_bands(_data, prefix="all_bands"):
    for i in range(_data.shape[2]):
        fig, ax = plt.subplots()
        ax.imshow(_data[:, :, i].squeeze())
        filename = os.path.join(SAVE_DIR, f"{prefix}_band_{i}.png")
        plt.savefig(filename)
        plt.close(fig)
        print(f"# Saved band {i} image: {filename}")


def plot_3d_data(_data, _colors=None, filename=None):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    _d = _data.reshape((-1, 3))
    ax.scatter3D(_d[:, 0], _d[:, 1], _d[:, 2], c=_colors)
    if filename is None:
        filename = os.path.join(SAVE_DIR, "3d_scatter.png")
    plt.savefig(filename)
    plt.close(fig)
    print(f"# Saved 3D scatter plot: {filename}")


def plot_spectra(_spectra, _bands, _legend=None, _areas=None, filename=None):
    fig = plt.figure()
    plt.xlabel("Wavelength")
    plt.ylabel("Reflectance")

    if _spectra.ndim == 1:
        plt.plot(_bands, _spectra)
        if _areas is not None:
            plt.fill_between(_bands, _areas.min, _areas.max)
    else:
        for _c, _s in enumerate(_spectra):
            plt.plot(_bands, _s)
            if _areas is not None:
                plt.fill_between(_bands, _areas[_c]['min'], _areas[_c]['max'], alpha=0.2)

    if _legend is not None:
        plt.legend(_legend)
    
    if filename is None:
        filename = os.path.join(SAVE_DIR, "spectra_plot.png")
    plt.savefig(filename)
    plt.close(fig)
    print(f"# Saved spectra plot: {filename}")


# --------------------- Image IO ---------------------

def write_array_image(_img, _name):
    """Write a grayscale image to file."""
    if _img.ndim > 2:
        _normalized_img = _img.reshape(-1, _img.shape[2])
        _normalized_img = (_normalized_img - _normalized_img.min(axis=0)) / \
            (_normalized_img - _normalized_img.min(axis=0)).max(axis=0)
        _normalized_img = _normalized_img.reshape(_img.shape)
        _img = _normalized_img
    Image.fromarray((_img.squeeze() * 255).astype(np.uint8)).save(_name)
    print(f"# Wrote grayscale image to: {_name}")


def load_image_array(_name):
    """Load a grayscale image from file."""
    img = Image.open(_name)
    print(f"# Loaded grayscale image from: {_name}")
    return img


# --------------------- Spectral & Camera Utilities ---------------------

def get_wavelengths_for(_c: CameraType):
    if _c == CameraType.VIS:
        return spectral_io.VIS_BANDS
    if _c == CameraType.VIS_COR:
        return spectral_io.VIS_COR_BANDS
    if _c == CameraType.NIR:
        return spectral_io.NIR_BANDS
    raise Exception("Unknown camera_type")


def get_camera_type_by_bands(bands: int):
    if bands == len(spectral_io.VIS_BANDS):
        return CameraType.VIS
    if bands == len(spectral_io.NIR_BANDS):
        return CameraType.NIR
    if bands == len(spectral_io.VIS_COR_BANDS):
        return CameraType.VIS_COR
    raise Exception("Unknown camera_type")


def get_random_spectra(_data, _number):
    _internal_data = _data.reshape(-1, _data.shape[-1])
    _idx = np.random.randint(0, _internal_data.shape[0], _number)
    return _internal_data[_idx]


def kmeans(_data, _num_of_clusters):
    _kmeans = KMeans(_num_of_clusters)
    _kmeans.fit(_data)
    return _kmeans.predict(_data)


# --------------------- Train/Test Split ---------------------

def split_into_train_and_test_fixed_and_evenly(_list, _ratio):
    if _ratio == 0:
        return _list, []
    if _ratio == 1:
        return [], _list

    record_count = len(_list)
    _labeled = np.array(_list)
    _validation_set_size = record_count * _ratio
    _each_nth_element = record_count / _validation_set_size

    _idx = []
    _i = 0.0
    while _i < record_count:
        _idx.append(int(_i))
        _i += _each_nth_element

    _mask = np.full(record_count, False, dtype=bool)
    _mask[_idx] = True
    _validation = _labeled[_mask]
    _train = np.array(_labeled[~_mask])
    return _train, _validation


def split_into_train_and_val(_list, _ratio):
    record_count = len(_list)
    _labeled = np.array(_list)
    _validation_set_size = int(record_count * _ratio)
    _idx = np.random.randint(record_count, size=_validation_set_size)
    _mask = np.full(record_count, False, dtype=bool)
    _mask[_idx] = True
    _validation = _labeled[_mask]
    _train = np.array(_labeled[~_mask])
    return _train, _validation


# --------------------- Memory Utilities ---------------------

def mem_report():
    print("### BEGIN - Memory Report ###")
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(f"# \t {type(obj)} : device {obj.device} with size {obj.size()}")
    print("### END - Memory Report ###")


class MemoryCheckpoint:
    def __init__(self):
        self.mem_report_checkpoint_objs = []

    def checkpoint(self):
        print("### Checkpoint - Memory Report ###")
        self.mem_report_checkpoint_objs = []
        for obj in gc.get_objects():
            if torch.is_tensor(obj):
                self.mem_report_checkpoint_objs.append(obj)

    def print_since_checkpoint(self):
        print("### BEGIN - Memory Report since Checkpoint ###")
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj not in self.mem_report_checkpoint_objs:
                print(f"# \t {type(obj)} : device {obj.device} with size {obj.size()}")
        print("### END - Memory Report ###")


# --------------------- Wandb Logging ---------------------

def get_wandb_log_dir():
    dir = os.environ.get('WANDB_LOG_DIR')
    if dir is not None:
        os.makedirs(dir, exist_ok=True)
    return dir
