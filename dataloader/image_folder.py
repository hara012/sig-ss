import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(path_files, dir=''):
    if path_files.find('.txt') != -1:
        paths, size = make_dataset_txt(path_files, dir)
    elif path_files.find('.csv') != -1:
        paths, size = make_dataset_csv(path_files, dir)
    elif path_files.find('.lst') != -1:
        paths, size = make_dataset_lst(path_files, dir)
    else:
        paths, size = make_dataset_dir(path_files)

    return paths, size

def make_dataset_txt(files, dir):
    """
    :param path_files: the path of txt file that store the image paths
    :return: image paths and sizes
    """
    img_paths = []

    with open(files) as f:
        paths = f.readlines()

    for path in paths:
        path = path.strip()
        img_paths.append(dir + path)

    return img_paths, len(img_paths)


def make_dataset_dir(dir):
    """
    :param dir: directory paths that store the image
    :return: image paths and sizes
    """
    img_paths = []

    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in os.walk(dir):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                img_paths.append(path)

    return img_paths, len(img_paths)

def make_dataset_csv(files, dir):
    """
    :param path_files: the path of txt file that store the image paths
    :return: image paths and sizes
    """
    img_paths = []

    with open(files) as f:
        for line in f:
            val = line.strip().split(",")
            #img_paths.append((val[0], np.float32(val[1]), np.float32(val[2]), np.float32(val[3])))
            #print(os.path.join(dir, val[0]))
            img_paths.append((os.path.join(dir, val[0]), np.float32(val[1:])))

    return img_paths, len(img_paths)

def make_dataset_lst(files, dir):
    """
    :param path_files: the path of txt file that store the image paths
    :return: image paths and sizes
    """
    img_paths = []

    with open(files) as f:
        paths = f.readlines()

    for path in paths:
        path = path.strip().split(',')
        img_paths.append(list(map(lambda s: dir + s, path)))
    
    return img_paths, len(img_paths)