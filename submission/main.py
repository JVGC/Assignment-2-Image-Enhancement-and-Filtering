## Name: João Victor Garcia Coelho  NUSP:   10349540
##       Paulo André de Oliveira Carneiro   10295304
## Course Code: SCC0251 - 2020/1


## Assignment 1 : Image Enhancement and Filtering

import numpy as np
import imageio

# Gaussian Function
def gaussian_kernel(x, sigma):
    return (1/(2*np.pi*(sigma**2)))*np.exp((-x**2)/(2*(sigma**2)))

## Calculating the Spatial Gaussian Component 
def spatial_gaussian_component(n, sigma):
    w = np.zeros([n, n]) # creating the kernel
    a = int((n-1)/2)
    for x in range(n):
        for y in range (n):
            ## Euclidian Distance between this pixel and the center pixel
            euclidian = np.sqrt((x-a)**2+ (y-a)**2)
            w[x,y] =  gaussian_kernel(euclidian, sigma)
            
    return w

def bilateral_filter(input_img, kernel_size, sigma_s, sigma_r):

    ## Adding padding zero in the input image
    nrows, ncols =  input_img.shape
    input_img =  np.concatenate((input_img, np.zeros([1, ncols])), axis=0)
    input_img =  np.concatenate((np.zeros([1, ncols]), input_img), axis=0)
    input_img =  np.concatenate((np.zeros([nrows+2, 1]), input_img), axis=1)
    input_img =  np.concatenate(( input_img, np.zeros([nrows+2, 1])), axis=1)

    a = int((n-1)/2)

    ## Creating the output image as a zero matriz
    output_img = np.zeros(input_img.shape, dtype=np.uint8)
    nrows, ncols = input_img.shape

    for x in range (a, nrows-a):
        for y in range (a, ncols-a):
            
            # for every pixel, the normalization factor (Wp)
            # and the the Intensity of the output pixel 
            # are initialized as 0
            Wp = 0 
            If = 0

            # gets subImage
            sub_img = input_img[ x-a : x+a+1 , y-a:y+a+1 ]

            ## Calculating the range gaussian for every neighboor of pixel (x,y)
            range_gaussian = gaussian_kernel(sub_img - input_img[x,y], sigma_r)

            ## Calculating the spatial component centered at the pixel (x,y)
            spatial_gaussian = spatial_gaussian_component(n, sigma_s)


            ## Final value of the filter
            w = np.multiply(range_gaussian, spatial_gaussian)

            ## Calculating the normalization factor by summing the filter's values
            Wp = np.sum(w)

            ## Multiplying the filter's value of each pixel by its intensity in the input image
            ## And summing all these values
            If = np.sum(np.multiply(w, sub_img))

            ## Finally, calculating the output value of the pixel (x,y)
            ## by dividing it by the normalization value
            output_img[x,y] =  If/Wp
    return input_img, output_img



filename = str(input()).rstrip() 
method = int(input()) 
save = int(input())

input_img = imageio.imread(filename)

if method == 1:
    n = int(input())
    sigma_s = float(input()) 
    sigma_r = float(input())
    input_img, output_img = bilateral_filter(input_img, n, sigma_s, sigma_r)

#ERROR
rse = np.sqrt(np.sum((input_img - output_img)**2))  

# Printing the error rounding to 4 decimal places.
print('{:.4f}'.format(rse))