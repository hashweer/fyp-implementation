import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms
import os


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def line_segment(base_path, test_path, pre_processed_path, image_name):
    print("Line Segmenting Started...")

    image_pre_processed = cv2.imread(base_path + pre_processed_path + image_name, 0)
    image_original = cv2.imread(base_path + test_path + image_name)

    height, width, channels = image_original.shape
    # if image size is larger, down sample
    if height > 1100 or width > 1100:
        image_original = cv2.pyrDown(image_original)

    image_pre_processed = 255 - image_pre_processed

    img_row_sum = np.sum(image_pre_processed, axis=1).tolist()

    # ignore counts less than 100
    for i in range(len(img_row_sum)):
        if img_row_sum[i] < 10000:
            img_row_sum[i] = 0

    cords = []
    # identify rising points and falling points
    for i in range(len(img_row_sum) - 1):

        current_value = img_row_sum[i]
        next_value = img_row_sum[i + 1]

        if current_value == 0 and next_value == 0:
            continue
        elif current_value > 0 and next_value > 0:
            continue
        elif current_value == 0 and next_value > 0:
            cords.append(i + 1)
        elif current_value > 0 and next_value == 0:
            cords.append(i)

    # transform graph
    base = plt.gca().transData
    rot = matplotlib.transforms.Affine2D().rotate_deg(270)
    #plt.plot(img_row_sum, transform=rot + base)
    #plt.show()

    H, W = image_pre_processed.shape[:2]
    differences = []

    # get line differences
    for i in range(len(cords) - 1):
        current_value = cords[i]
        next_value = cords[i + 1]

        difference = next_value - current_value

        # filter values
        if 10 < difference < 120:
            differences.append(difference)
        else:
            differences.append(0)

    # detect pairs
    pairs = []

    for i in range(len(differences)):

        pair = []
        if 15 < differences[i] < 80:
            pair.append(cords[i])
            pair.append(cords[i+1])
            pairs.append(pair)

    image_pre_processed = 255 - image_pre_processed

    line_segment_base_path = 'step_9_vertical_projection_filtered/'
    image_name_array = image_name.split('.')
    line_segmented_path = line_segment_base_path + 'segmented/' + image_name_array[0]

    if not os.path.exists(base_path + line_segmented_path):
        os.makedirs(base_path + line_segmented_path)

    for i in range(len(pairs)):
        pair = pairs[i]

        roi = image_pre_processed[pair[0]:pair[1], 0:W]
        # count black pixels
        black_pixel_count = np.sum(roi == 0)

        # filter segments by black pixel count
        if black_pixel_count > 100:
            cv2.line(image_original, (0, pair[0]), (W, pair[0]), (0, 0, 255), 1)
            cv2.line(image_original, (0, pair[1]), (W, pair[1]), (0, 255, 0), 1)

            roi = image_resize(roi, height=40)
            cv2.imwrite(os.path.join(base_path + line_segmented_path, str(i) + '.' + image_name_array[1]), roi)

    cv2.imshow('Segmented Image', image_original)
    cv2.imwrite(os.path.join(base_path + line_segment_base_path + image_name), image_original)
    #cv2.waitKey(0)

    print("Line Segmenting Done.")

