# Datasets
## MNIST
The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.

### Download
Please unzip and place `train.pt` and `test.pt` in `$DATA_ROOT/processed` folder.

- [Google Drive](https://drive.google.com/open?id=0B0AsKkiz_kZRLXRZbGxIX2t3QTA)

## MNIST-M
The MNIST-M dataset consists of MNIST digits blended with random color patches from the BSDS500 dataset. Its structure is the same as MNIST.

### Download
Please unzip and place `mnist_m_train.pt` and `mnist_m_test.pt` in `$DATA_ROOT/processed` folder.

- [Google Drive](https://drive.google.com/open?id=0B0AsKkiz_kZRZUxMNW5YTlkzOUk)

## USPS
The dataset refers to numeric data obtained from the scanning of handwritten digits from envelopes by the U.S. Postal Service. The original scanned digits are binary and of different sizes and orientations; the images here have been deslanted and size normalized, resulting in 16 x 16 grayscale images (Le Cun et al., 1990).

There are 7291 training observations and 2007 test observations, distributed as follows:

|       | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | Total |
| ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- |
| Train | 1194 | 1005 | 731  | 658  | 652  | 556  | 664  | 645  | 542  | 644  | 7291  |
| Test  | 359  | 264  | 198  | 166  | 200  | 160  | 170  | 147  | 166  | 177  | 2007  |

### Download
Please place `usps_28x28.pkl` in `$DATA_ROOT` folder.
- [GitHub](https://raw.githubusercontent.com/mingyuliutw/CoGAN_PyTorch/master/data/uspssample/usps_28x28.pkl)
- [Google Drive](https://drive.google.com/open?id=0B0AsKkiz_kZRNy11MHE3Q01CdEk)
