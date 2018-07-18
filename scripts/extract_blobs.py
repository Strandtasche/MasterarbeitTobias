"""
This script serves to read images given the  provided input-directory (if none provided,
 it uses the current directory) and given the file-extension of the images. After reading the images
 it subtracts the background and calculates the blobs (= particles). Those are extracted afterwards
 and saved in the provided output-directory.

Additionally, important data is saved into a csv-file.

 Parameters:
     - input_directory: the directory which contains the images (if none: current_directory); path is relative
     - file_extension: file-extension of the images which shall be looked for, other files are ignored
     - output_directory: directory the extracted blobs shall be saved to
"""

from scipy import misc
from scipy import ndimage
import glob as glob
import numpy as np
import os
import argparse
import datetime
import csv

# brief: reads the images
# path: relative to current folder
# file_extension: extensions of the file
# return: list of images as np.array
def read_images (path, file_extension):
    path_name = path + '/*' + file_extension

    # NOTE: glob collects files randomly (or at least in an arbitrary order) --> needs to be sorted alphabetically
    file_list = glob.glob(path_name)
    image_list = [np.array(misc.imread(file)) for file in sorted(file_list)]
    num_images = len(image_list)
    last_index = 0
    image_batches = list()
    batch_size = 100

    # split the list into batches to avoid memory error
    while last_index < num_images:
        temp_batch_size = batch_size
        if batch_size > num_images - last_index:
            temp_batch_size = num_images - last_index
        image_batches.append(np.array(image_list[last_index:last_index+temp_batch_size]))
        last_index += temp_batch_size
    #image_batches = np.array(image_batches)

    print("Finished reading {} images.".format(num_images))
    return image_batches, num_images

# brief: subtracts the background
# images: input images
# threshold: threshold between background/foreground
# return: list of binarized images
def subtract_bg(images, threshold=30):
    THRESHOLD = threshold
    ret_images = np.array(images)
    ret_images[np.absolute(ret_images - np.mean(ret_images, axis=0)) <= THRESHOLD] = 0
    ret_images[ret_images > 0] = 1

    return ret_images

# brief: labels the images
# binarized_images: images already binarized in background/foreground
# threshold: minimum required connected pixels for a particle to be recognized
# return: list of labeled images
def label_images(binarized_images, threshold):
    ret_images = np.array(binarized_images)

    index = 0
    for image in binarized_images:
        image, num_particles = ndimage.label(image)

        # ensures there are at least THRESHOLD connected pixels
        for i in range(0, num_particles + 1):
            if np.count_nonzero(image[image == i]) <= threshold:
                image[image == i] = 0

        ret_images[index] = image # save new picture
        index = index+1

    labeled_images, num_particles = ndimage.label(ret_images)

    return labeled_images

# brief: saves the blobs in the given directory
# blobs: list of coordinates for the blob-boxes
# main image: image in which the blobs can be found
# dir: relative path to current folder
# return:
def save_blobs(blobs, main_image, dir, csv_writer, timestep, greyscale, height):

    if not os.path.exists(dir) and not args.noimage:
        os.makedirs(dir)

    count = 1

    csv_row_data = list()
    csv_row_data.append(timestep)

    # following slice needs to defined for coloured images
    channel_slice = slice(0, 3, None)

    # loop over all the blobs
    for iter_blobs, blob_box_coordinates in enumerate(blobs):

        if blob_box_coordinates is not None:

            if not greyscale:
                img = main_image[blob_box_coordinates[0], blob_box_coordinates[1], channel_slice]  # retrieve image-blob
            else:
                img = main_image[blob_box_coordinates[0], blob_box_coordinates[1]]

            # calculate "centroids" of blobs
            y_slice = blob_box_coordinates[0]
            x_slice = blob_box_coordinates[1]

            x_start = x_slice.start
            x_end = x_slice.stop

            y_start = y_slice.start
            y_end = y_slice.stop

            csv_row_data.append((x_end + x_start)/2)
            csv_row_data.append(height - (y_end + y_start)/2)

            if not args.noimage:
                merged_path = os.path.join(dir, "blob_" + str(count) + ".png")
                misc.imsave(merged_path, img)  # save image
            count += 1
    csv_row_data.insert(1, count-1)
    csv_writer.writerow(csv_row_data)

if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--input_directory',
                        help='An argument for the input directory')

    parser.add_argument('--file_extension',
                        help='File extension of the input pictures')

    parser.add_argument('--output_directory',
                        help='An argument for the output directory')
    parser.add_argument('--noimage', help="ignore image output and only export csv", action="store_true" )

    args = parser.parse_args()
    if args.input_directory == None:
        print('Input directory path is required!')
        print('Taking current working directory instead')
        args.input_directory = os.getcwd()

    elif args.file_extension == None:
        print('File extension is required!')
        #quit()

    elif args.output_directory == None:
        print('Output directory path is required!')
        quit()

    if not os.path.exists(args.input_directory):
        print (args.input_directory + " path does not exist!")
    #    quit()

    print ("Input Directory: " + args.input_directory)
    print ("Output Directory: " + args.output_directory)

    print("######################################################")
    print("STARTED: {}".format(datetime.datetime.now().time()))
    print("######################################################")

    images, num_images = read_images(args.input_directory, args.file_extension)

    # get image dimensioons
    # print(images[0][0].shape)
    HEIGHT, WIDTH, channels = images[0][0].shape

    # subtract bg
    images_without_bg = list()
    counter = 0
    outputVals = [i/10 * len(images) for i in range(11)]
    for batch in images:
        images_without_bg.append(subtract_bg(batch))
        if counter in outputVals:
            print("Progress subtracting background: {}%".format(int(100 * counter/len(images))))
        counter = counter + 1
    print("Finished subtracting background.")

    # label batches
    images_labeled = list()
    counter = 0
    outputVals = [i/10 * len(images) for i in range(11)]
    for batch in images_without_bg:
        images_labeled.append(label_images(batch, threshold=200))
        if counter in outputVals:
            print("Progress labeling images: {}%".format(int(100 * counter/len(images))))
        counter = counter + 1
    print("Finished labeling images.")

    # now flatten the lists
    images_labeled_list_flattened = []
    for batch in images_labeled:
        for image in batch:
            images_labeled_list_flattened.append(image)

    image_list_flattened = []
    for batch in images:
        for image in batch:
            image_list_flattened.append(image)
    print("Finished flattening batches.")

    # create csv file and write header
    csvfile = open(args.output_directory + '.csv', 'w')

    writer = csv.writer(csvfile, delimiter=',')

    timestep = 1 # iterator over the images

    # check if image is greyscale or not
    greyscale = True

    # loop over all images
    for curr_image_labeled, image in zip(images_labeled_list_flattened, image_list_flattened):

        blobs = ndimage.find_objects(curr_image_labeled)

        output_dir = os.path.join(args.output_directory, "image" + str(timestep)) # name the directory according to the image-id
        save_blobs(blobs, main_image=image, dir=output_dir, csv_writer=writer, timestep=timestep, greyscale=greyscale, height=HEIGHT)
        print("Done with extracting blobs for picture {0} out of {1} pictures in total.".format(timestep, num_images))
        timestep += 1
    # now close csv file
    csvfile.close()

    # now add header which is important for postprocessing
    non_nan_list = list()
    with open(args.output_directory + '.csv', 'r') as read_file:
        csvreader = csv.reader(read_file)

        for row in csvreader:
            non_nan_list.append(row)

    max_length = max(len(row) for row in non_nan_list)

    # fill up empty cells with nans
    nan_filled_up_list = list()
    for row in non_nan_list:
        row += ['NaN'] * (max_length - len(row))
        nan_filled_up_list.append(row)

    # create the header
    header = list()
    header.append('FrameNr')
    header.append('NumberMidPoints')

    for i in range(1, int((max_length-2)/2 + 1)):
        header.append('MidPointX_' + str(i))
        header.append('MidPointY_' + str(i))

    # overwrite it
    with open(args.output_directory + '.csv', 'w') as write_file:
        csvwriter = csv.writer(write_file)
        csvwriter.writerow(header)
        csvwriter.writerows(nan_filled_up_list)

    print("######################################################")
    print("FINISHED: {}".format(datetime.datetime.now().time()))
    print("######################################################")
