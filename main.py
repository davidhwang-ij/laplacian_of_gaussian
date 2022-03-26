# import dependencies
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# open .img file, remove headers, and save as png
with open("./test1.img", 'rb') as f:
    f.seek(512)
    test1_img = np.fromfile(f, dtype=np.uint8).reshape((512, 512))

plt.imsave('test1.png', test1_img)

with open("./test2.img", 'rb') as f:
    f.seek(512)
    test2_img = np.fromfile(f, dtype=np.uint8).reshape((512, 512))

plt.imsave('test2.png', test2_img)

with open("./test3.img", 'rb') as f:
    f.seek(512)
    test3_img = np.fromfile(f, dtype=np.uint8).reshape((512, 512))

plt.imsave('test3.png', test3_img)

test1 = cv.imread('test1.png', 0)
test2 = cv.imread('test2.png', 0)
test3 = cv.imread('test3.png', 0)


# Generate the Gaussian kernel
def generate_kernel(sigma, size):

    # Generate initial kernel grids
    x, y = np.meshgrid(np.arange(-size/2+1, size/2+1),
                       np.arange(-size/2+1, size/2+1))

    # LoG operator
    kernel = ((x**2 + y**2 - (2*sigma**2)) / sigma**4) * \
        np.e**(-(x**2+y**2) / (2.0*sigma**2))

    # Normalize
    kernel *= 1.0/kernel.max()

    return kernel


# Perform convolution
def convolve(img, sigma):

    # Define kernel size and generate the gaussian kernel
    size = int(2*(np.ceil(3*sigma))+1)
    kernel = generate_kernel(sigma, size)

    # Save the shapes of the image and the kernel
    img_row, img_col = img.shape
    kernel_size = kernel.shape[0]
    padding_size = int((kernel_size - 1) // 2)

    # Add padding to the image
    padded_img = np.zeros((img_row + (2 * padding_size),
                           img_col + (2 * padding_size)))
    padded_img[padding_size:padded_img.shape[0] - padding_size,
               padding_size:padded_img.shape[1] - padding_size] = img

    # Convolve with the Gaussian kernel and store the output
    output = np.zeros(img.shape)
    for row in range(img_row-1):
        for col in range(img_col-1):
            output[row, col] = np.sum(
                kernel * padded_img[row:row + kernel_size, col:col + kernel_size])

    return output


def generate_edge_map(out):
    row = out.shape[0]
    col = out.shape[1]
    edge_map = np.zeros(out.shape)

    # Compute zero-crossing
    for i in range(1, row-1):
        for j in range(1, col-1):
            if out[i][j] == 0:
                if ((out[i-1][j-1] < 0 and out[i+1][j+1] > 0) or (out[i-1][j-1] > 0 and out[i+1][j+1] < 0) or
                    (out[i-1][j+1] < 0 and out[i+1][j-1] > 0) or (out[i-1][j+1] > 0 and out[i+1][j-1] < 0) or
                        (out[i][j-1] < 0 and out[i][j+1] > 0) or (out[i][j-1] > 0 and out[i][j+1] < 0) or
                        (out[i-1][j] < 0 and out[i+1][j] > 0) or (out[i-1][j] > 0 and out[i+1][j] < 0)):
                    edge_map[i][j] = 1
            elif out[i][j] < 0:
                if ((out[i-1][j-1] > 0) or (out[i-1][j+1] > 0) or
                    (out[i+1][j-1] > 0) or (out[i+1][j+1] > 0) or
                    (out[i][j-1] > 0) or (out[i][j+1] > 0) or
                        (out[i-1][j] > 0) or (out[i+1][j] > 0)):
                    edge_map[i][j] = 1

    return edge_map


def generate_new_edge_map(out, edge_map):
    row = out.shape[0]
    col = out.shape[1]
    new_edge_map = np.zeros(out.shape)

    # Compute zero-crossing where where E(x,y; sigma) = 1 and the 8-neighbors of these pixel locations
    for i in range(1, row-1):
        for j in range(1, col-1):

            # Compute at pixel locations where E(x,y; sigma) = 1
            if edge_map[i][j] == 1:
                if out[i][j] == 0:
                    if ((out[i-1][j-1] < 0 and out[i+1][j+1] > 0) or (out[i-1][j-1] > 0 and out[i+1][j+1] < 0) or
                        (out[i-1][j+1] < 0 and out[i+1][j-1] > 0) or (out[i-1][j+1] > 0 and out[i+1][j-1] < 0) or
                        (out[i][j-1] < 0 and out[i][j+1] > 0) or (out[i][j-1] > 0 and out[i][j+1] < 0) or
                            (out[i-1][j] < 0 and out[i+1][j] > 0) or (out[i-1][j] > 0 and out[i+1][j] < 0)):
                        new_edge_map[i][j] = 1
                elif out[i][j] < 0:
                    if ((out[i-1][j-1] > 0) or (out[i-1][j+1] > 0) or
                        (out[i+1][j-1] > 0) or (out[i+1][j+1] > 0) or
                        (out[i][j-1] > 0) or (out[i][j+1] > 0) or
                            (out[i-1][j] > 0) or (out[i+1][j] > 0)):
                        new_edge_map[i][j] = 1

    return new_edge_map


def multiscale_log(img, sigma=5.0):
    while sigma >= 1.0:

        # Perform convolution on the image with the provided sigma value
        output = convolve(img, sigma)
        output = output.astype(np.int64, copy=False)

        edge_map = generate_edge_map(output)

        # Display gray scale image and edge maps for sigma=5.0, 4.0, 3.0, 2.0, and 1.0
        if float(sigma).is_integer():
            plt.imshow(output, cmap='gray')
            plt.imshow(edge_map, cmap='gray')
            plt.figure(int(sigma)+1)

        sigma_new = sigma - 0.5

        # Generate edge map with the new sigma
        output_new = convolve(img, sigma_new)
        output_new = output_new.astype(np.int64, copy=False)
        edge_map_new = generate_new_edge_map(output_new, edge_map)

        # Replace edge map and sigma with the new edge map and sigma
        edge_map = edge_map_new
        sigma = sigma_new

    plt.show()


if __name__ == "__main__":
    multiscale_log(test1)
    # multiscale_log(test2)
    # multiscale_log(test3)
