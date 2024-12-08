import sys
from PIL import Image
import warp as wp
import numpy as np

wp.init()
device = "cpu"

@wp.kernel #Unsharp masking for greyscale images
def greyscaleSharpen(input: wp.array(dtype=wp.float32, ndim=2), output: wp.array(dtype=wp.float32, ndim=2), constant: float, kernelSize: int):
    i, j = wp.tid() #Keeps track of two tid values (One for each dimension of the array)
    distanceFromCenter = kernelSize // 2 #Distance from the center of the matrix (Used to iterate over n*n sized matrices)
    matrixSum = float(0.0) #Keeping track of the sum of all values within the n*n matrix
    nums = float(0) #Keeping track of the number of values within the n*n matrix

    output[i, j] = input[i, j]

    for iIndex in range(-distanceFromCenter, distanceFromCenter + 1): #Iterating over all values in an n*n matrix
        for jIndex in range(-distanceFromCenter, distanceFromCenter + 1):
            if i+iIndex >= 0 and i+iIndex < input.shape[0] and j+jIndex >= 0 and j+jIndex < input.shape[1]: #Option two of border handling: Kernel Truncation (Checking to see if the pixel is within the bounds of the image)
                matrixSum += input[i+iIndex, j+jIndex]
                nums += 1.0

    output[i, j] = matrixSum / nums #Finding average of all values in the n*n matrix
    output[i, j] = input[i, j] - output[i, j] #Subtracting the average (blurred) pixel from the original pixel
    output[i, j] = input[i, j] + (constant * (output[i, j])) #Adding the previous calculation multiplied by the constant to the original image

@wp.kernel #Unsharp masking for RGB colour images
def colourSharpen(input: wp.array(dtype=wp.float32, ndim=3), output: wp.array(dtype=wp.float32, ndim=3), constant: float, kernelSize: int):
    i, j, k = wp.tid() #Keeps track of three tid values (One for each dimension of the array)
    distanceFromCenter = kernelSize // 2 #Distance from the center of the matrix (Used to iterate over n*n sized matrices)
    matrixSum = float(0.0) #Keeping track of the sum of all values within the n*n matrix
    nums = float(0) #Keeping track of the number of values within the n*n matrix

    output[i, j, k] = input[i, j, k]

    for iIndex in range(-distanceFromCenter, distanceFromCenter + 1): #Iterating over all values in an n*n matrix
        for jIndex in range(-distanceFromCenter, distanceFromCenter + 1):
            if i+iIndex >= 0 and i+iIndex < input.shape[0] and j+jIndex >= 0 and j+jIndex < input.shape[1]: #Option two of border handling: Kernel Truncation (Checking to see if the pixel is within the bounds of the image)
                matrixSum += input[i+iIndex, j+jIndex, k]
                nums += 1.0

    output[i, j, k] = matrixSum / nums #Finding average of all values in the n*n matrix
    output[i, j, k] = input[i, j, k] - output[i, j, k] #Subtracting the average (blurred) pixel from the original pixel
    output[i, j, k] = input[i, j, k] + (constant * (output[i, j, k])) #Adding the previous calculation multiplied by the constant to the original image

@wp.kernel #Median filtering for greyscale images
def greyscaleNoise(input: wp.array(dtype=wp.float32, ndim=2), output: wp.array(dtype=wp.float32, ndim=2), kernelSize: int):
    i, j = wp.tid() #Keeps track of two tid values (One for each dimension of the array)
    distanceFromCenter = kernelSize // 2 #Distance from the center of the matrix (Used to iterate over n*n sized matrices)
    nums = float(0) #Keeping track of the number of values within the n*n matrix
    v = wp.vector(dtype=float, length=200) #Local storage to hold an "array" of values in matrix

    for iIndex in range(-distanceFromCenter, distanceFromCenter + 1): #Iterating over all values in an n*n matrix
        for jIndex in range(-distanceFromCenter, distanceFromCenter + 1):
            if i+iIndex >= 0 and i+iIndex < input.shape[0] and j+jIndex >= 0 and j+jIndex < input.shape[1]: #Option two of border handling: Kernel Truncation (Checking to see if the pixel is within the bounds of the image)
                v[int(nums)] = input[i+iIndex, j+jIndex]
                nums += 1.0

    for num in range(int(nums) - 1, 0, -1): #Simple bubble sort on the vector that holds all of the matrix values (To get median of a list of numbers, the numbers have to be sorted)
        for swap in range(0, num):
            if v[swap] > v[swap + 1]: #Swap values
                temp = v[swap]
                v[swap] = v[swap + 1]
                v[swap + 1] = temp

    median = v[int(nums) // 2] #Finding median value of the matrix values

    output[i, j] = median #Setting the pixel to the value of the median of the n*n matrix


@wp.kernel #Median filtering for RGB colour images
def colourNoise(input: wp.array(dtype=wp.float32, ndim=3), output: wp.array(dtype=wp.float32, ndim=3), kernelSize: int):
    i, j, k = wp.tid() #Keeps track of two tid values (One for each dimension of the array)
    distanceFromCenter = kernelSize // 2 #Distance from the center of the matrix (Used to iterate over n*n sized matrices)
    nums = float(0) #Keeping track of the number of values within the n*n matrix
    v = wp.vector(dtype=float, length=200) #Local storage to hold an "array" of values in matrix

    for iIndex in range(-distanceFromCenter, distanceFromCenter + 1): #Iterating over all values in an n*n matrix
        for jIndex in range(-distanceFromCenter, distanceFromCenter + 1):
            if i+iIndex >= 0 and i+iIndex < input.shape[0] and j+jIndex >= 0 and j+jIndex < input.shape[1]: #Option two of border handling: Kernel Truncation (Checking to see if the pixel is within the bounds of the image)
                v[int(nums)] = input[i+iIndex, j+jIndex, k]
                nums += 1.0

    for num in range(int(nums) - 1, 0, -1): #Simple bubble sort on the vector that holds all of the matrix values (To get median of a list of numbers, the numbers have to be sorted)
        for swap in range(0, num):
            if v[swap] > v[swap + 1]: #Swap values
                temp = v[swap]
                v[swap] = v[swap + 1]
                v[swap + 1] = temp

    median = v[int(nums) // 2] #Finding median value of the matrix values

    output[i, j, k] = median #Setting the pixel to the value of the median of the n*n matrix

if len(sys.argv) != 6: #Exit program if user didn't input correct number of command line arguements
    sys.exit("Incorrect number of command line arguements. The program needs 6 command line arguements.")

image = Image.open(sys.argv[4]) #File name is 5th command line arguement

numpyData = np.asarray(image, dtype='float32') #Creating a numpy array from the contents of the image

inWarpImage = wp.from_numpy(numpyData, dtype=wp.float32, device=device) #Creasting input array for kernels
outWarpImage = wp.zeros(shape=numpyData.shape, dtype=wp.float32, device=device) #Creating output array for kernels (all values initialized to 0)

if sys.argv[1] == "-s" and image.mode == "L": #If image is greyscale and sharpen command line arguement passed
    constant = wp.constant(float(sys.argv[3])) #Passing constant for unsharp masking algorithm to kernel
    kernelSize = wp.constant(int(sys.argv[2])) #Passing kernelSize as an arguement to kernel

    wp.launch(kernel=greyscaleSharpen, #Launching kernel
              dim=numpyData.shape,
              inputs=[inWarpImage, outWarpImage, constant, kernelSize],
              device=device)

elif sys.argv[1] == "-s" and image.mode == "RGB": #If image is RGB coloured and sharpen command line arguement passed
    constant = wp.constant(float(sys.argv[3])) #Passing constant for unsharp masking algorithm to kernel
    kernelSize = wp.constant(int(sys.argv[2])) #Passing kernelSize as an arguement to kernel
    
    wp.launch(kernel=colourSharpen, #Launching kernel
              dim=numpyData.shape,
              inputs=[inWarpImage, outWarpImage, constant, kernelSize],
              device=device)

elif sys.argv[1] == "-n" and image.mode == "L": #If image is greyscale and noise reduction command line arguement passed
    kernelSize = wp.constant(int(sys.argv[2])) #Passing kernelSize as an arguement to kernel

    if kernelSize > 13:
        sys.exit("Error: Kernel size too big, noise reduction algorithm only supports kernels of size 13 or smaller")

    wp.launch(kernel=greyscaleNoise, #Launching kernel
              dim=numpyData.shape,
              inputs=[inWarpImage, outWarpImage, kernelSize],
              device=device)

elif sys.argv[1] == "-n" and image.mode == "RGB": #If image is RGB coloured and noise reduction command line arguement passed
    kernelSize = wp.constant(int(sys.argv[2])) #Passing kernelSize as an arguement to kernel

    if kernelSize > 13:
        sys.exit("Error: Kernel size too big, noise reduction algorithm only supports kernels of size 13 or smaller")

    wp.launch(kernel=colourNoise, #Launching kernel
              dim=numpyData.shape,
              inputs=[inWarpImage, outWarpImage, kernelSize],
              device=device)

numpyOutArray = np.asarray(wp.array.numpy(outWarpImage), dtype='float32') #Turns output warp array into numpy array

sharpened_clipped = np.clip(numpyOutArray, 0.0, 255.0) #Removes extra purple/blue colours in image

imageOut = Image.fromarray(np.uint8(sharpened_clipped)) #Creating an image from the values in the numpy array

imageOut.save(sys.argv[5]) #Saving image as filename passed as 6th command line arguement