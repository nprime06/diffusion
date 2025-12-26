import numpy as np
import torch
from torch.utils.data import Dataset
import struct


_IDX_DTYPE_MAP = {
    0x08: np.uint8,
    0x09: np.int8,
    0x0B: np.int16,
    0x0C: np.int32,
    0x0D: np.float32,
    0x0E: np.float64,
}


def read_idx(path: str) -> np.ndarray:
    """
    Read an IDX file (e.g. MNIST .idx3-ubyte / .idx1-ubyte) into a numpy array.
    Spec: http://yann.lecun.com/exdb/mnist/
    """
    with open(path, "rb") as f:
        header = f.read(4)
        if len(header) != 4:
            raise ValueError(f"Invalid IDX file (too short): {path}")

        zero0, zero1, dtype_code, ndim = struct.unpack(">BBBB", header)
        if zero0 != 0 or zero1 != 0:
            raise ValueError(f"Invalid IDX magic (missing leading zeros): {path}")
        if dtype_code not in _IDX_DTYPE_MAP:
            raise ValueError(f"Unsupported IDX dtype code 0x{dtype_code:02x} in {path}")
        if ndim <= 0:
            raise ValueError(f"Invalid IDX ndim={ndim} in {path}")

        shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        data = f.read()

    arr = np.frombuffer(data, dtype=_IDX_DTYPE_MAP[dtype_code])
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(f"IDX size mismatch in {path}: expected {expected}, got {arr.size}")
    return arr.reshape(shape)

def load_mnist_images_labels(image_path, label_path, ):
    images = torch.from_numpy(read_idx(image_path).copy()).float()  # copy to avoid modifying the original data
    labels = torch.from_numpy(read_idx(label_path).copy()).long()
    return images, labels

class MNISTDataloader(Dataset): # everything is on cpu
    def __init__(self, test_images_path, test_labels_path, train_images_path, train_labels_path):
        self.test_images, self.test_labels = load_mnist_images_labels(test_images_path, test_labels_path)
        self.train_images, self.train_labels = load_mnist_images_labels(train_images_path, train_labels_path)
        self.images = torch.cat((self.test_images, self.train_images)) # (N, 28, 28)
        self.labels = torch.cat((self.test_labels, self.train_labels))

        self.images_mean = self.images.mean(dim=0, keepdim=True).float()
        self.images_std = self.images.std(dim=0, keepdim=True).float()
        self.images = (self.images - self.images_mean) / (self.images_std + 1e-6).float()
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx] # (28, 28), (1,)
    
    def get_mean_std(self):
        return self.images_mean, self.images_std # (1, 28, 28), (1, 28, 28)