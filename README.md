# Image Filters


## Program and Algorithm Explanations

In my Python program a3.py, I implemented four image processing kernels which
apply two different image filters, unsharp masking and median filtering, to either a
grayscale image or a RGB colour image.

## Running the a3.py file

In order to run the a3.py file, there are a few steps you need to take first before you
can run the file and produce a sharpened/noise reduced image. Firstly, you need to make
sure that you have all of the necessary dependencies installed for the program. In my
program, I make use of the Pillow, Warp and Numpy libraries, so you need to have those
three dependencies installed before attempting to run the program. Secondly, once
you download the a3.py file, you will want to place that Python file as
well as any images that you would like to filter in a directory on your computer that is
easy to navigate to. Once that's done, you can open up a terminal and navigate to the
directory containing the Python file and images and run the command:

python3 a3.py algType kernSize param inFileName outFileName

This command will take five command line arguments. Each of them has an important
use to the program.
- algType: Determines the filtering kernel that will be run on your image (-s for
unsharp masking, -n for median filtering)
- kernSize: The size of the n*n matrix that operates on each pixel (If using the noise
reduction kernels, the max size is 13*13, anything higher will alert the user and
exit the program)
- param: The constant used in the unsharp masking kernels (For the noise reduction
kernels, this value can be anything)
- inFileName: Name of the file you wish to apply the filter to
- outFileName: Name of the file that contains the image after the filters were
applied

## Important Note About Parallelism

In my program, I set the device to run the Warp kernels on to be the cpu, so the program will
not benefit from parallelism by default. To fix this, change the line in the code that sets the "device" variable
to "cuda" instead of "cpu". Make sure that you have a compadible Nvidia GPU before doing this.
