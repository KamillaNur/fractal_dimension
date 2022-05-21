import numpy as np
import imageio.v2 as imageio


# make the image black and white
def white_black(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    white_and_black = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return white_and_black


# find out the number of the image on which the finger is located
def pieces_finger_count(z, k):
    s = np.add.reduceat(np.add.reduceat(z, np.arange(0, z.shape[0], k), axis=0), np.arange(0, z.shape[1], k), axis=1)
    return len(np.where((s > 0) & (s < k * k))[0])


# fractal dimension
def fractal_dimension(z):
    # min dimension of image (weight or height)
    p = min(z.shape)
    # max power of 2 that does not exceed p
    n = 2 ** np.floor(np.log(p) / np.log(2))
    # extract the exponent
    n = int(np.log(n) / np.log(2))
    # size of the squares into which the image will be divided
    sizes = 2 ** np.arange(n, 0, -1)
    # find out pieces_finger_count()
    pieces = []
    for size in sizes:
        pieces.append(pieces_finger_count(z, size))
    coeffs = np.polyfit(np.log(sizes), np.log(pieces), 1)
    return -coeffs[0]


# read image
img1 = imageio.imread('finger1.png')
img2 = imageio.imread('finger2.png')

# convert image to black&white
bw1 = white_black(img1)
bw2 = white_black(img2)

# convert black&white image to binary array
bw1 = (bw1 < 0.9)
bw2 = (bw2 < 0.9)

finger1 = fractal_dimension(bw1)
finger2 = fractal_dimension(bw2)

print('fractal dimension:', finger1)
print('fractal dimension:', finger2)
