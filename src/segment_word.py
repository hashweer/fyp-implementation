import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


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


def word_segment(base_path, line_segments_path, image_name):
    print("Word Segmenting Started...")

    base_image_name_array = image_name.split('.')
    path = base_path + line_segments_path + base_image_name_array[0] + '/'

    line_list = []
    for (dir_path, dir_names, file_names) in os.walk(path):
        line_list.extend(file_names)
        break

    sorted_line_list = []

    for i in range(len(line_list)):
        image_name = line_list[i]
        image_name_array = image_name.split('.')
        sorted_line_list.append(int(image_name_array[0]))

    sorted_line_list.sort()

    word_segment_base_path = '../data/testimages/step_10_word_segment/'
    word_segmented_path = word_segment_base_path + 'segmented/' + base_image_name_array[0]

    if not os.path.exists(base_path + word_segmented_path):
        os.makedirs(base_path + word_segmented_path)

    for i in range(len(sorted_line_list)):

        line_segment_original = cv2.imread(path + str(sorted_line_list[i]) + '.' + base_image_name_array[1])
        # line_segment_original = image_resize(line_segment_original, height=40)
        line_segment = cv2.imread(path + str(sorted_line_list[i]) + '.' + base_image_name_array[1], 0)
        # line_segment = image_resize(line_segment, height=40)

        line_segment = 255 - line_segment
        line_segment_sum = np.sum(line_segment, axis=0).tolist()

        # print(line_segment_sum)


        # ignore counts less than 100
        for x in range(len(line_segment_sum)):
            if line_segment_sum[x] < 100:
                line_segment_sum[x] = 0

        # identify rising points and falling points
        cords = []
        for x in range(len(line_segment_sum)-1):
            current_value = line_segment_sum[x]
            next_value = line_segment_sum[x + 1]

            if current_value == 0 and next_value == 0:
                continue
            elif current_value > 0 and next_value > 0:
                continue
            elif current_value == 0 and next_value > 0:
                cords.append(x + 1)
            elif current_value > 0 and next_value == 0:
                cords.append(x)

        H, W = line_segment.shape[:2]

        # print(cords)

        differences = []

        # get line differences
        for x in range(len(cords) - 1):
            current_value = cords[x]
            next_value = cords[x + 1]

            difference = next_value - current_value

            # filter values
            if 10 < difference < 250:
                differences.append(difference)
            else:
                differences.append(0)

        # print(differences)

        # detect pairs
        pairs = []
        for x in range(len(differences)):

            pair = []
            if 10 < differences[x] < 250:
                pair.append(cords[x])
                pair.append(cords[x + 1])
                pairs.append(pair)

        line_segment = 255 - line_segment

        # print(pairs)
        # detect word spaces
        if len(pairs) > 0:
            word_cords = [pairs[0][0]]
        else:
            word_cords = [0]

        for x in range(len(pairs)):

            pair = pairs[x]

            roi = line_segment[0:H, pair[0]:pair[1]]

            # count black pixels
            black_pixel_count = np.sum(roi == 0)

            if black_pixel_count < 20:
                word_cords.append(pair[0])
                word_cords.append(pair[1])

        if len(pairs) > 0:
            last_cord = pairs[len(pairs) - 1][1]
            if W - last_cord > 5:
                word_cords.append(last_cord + 2)
            else:
                word_cords.append(W)
        else:
            word_cords.append(W)

        # print(word_cords)
        # arrange new pairs
        arranged_pairs = []
        for x in range(0, len(word_cords), 2):
            pair = [word_cords[x], word_cords[x+1]]
            arranged_pairs.append(pair)

        # draw new lines
        for x in range(len(arranged_pairs)):
            pair = arranged_pairs[x]
            roi = line_segment[0:H, pair[0]:pair[1]]

            cv2.line(line_segment_original, (pair[0], 0), (pair[0], H), (0, 0, 255), 1)
            cv2.line(line_segment_original, (pair[1], 0), (pair[1], H), (255, 0, 0), 1)

            cv2.imwrite(os.path.join(base_path + word_segmented_path, str(i+1) + '_' + str(x+1) + '.' + base_image_name_array[1]), roi)

        cv2.imwrite(os.path.join(base_path + word_segmented_path + (base_image_name_array[0] + str(i+1) + '.' + base_image_name_array[1])), line_segment_original)
        # cv2.imshow('segmented_words', line_segment_original)
        # plt.plot(line_segment_sum)
        # plt.show()
        # cv2.waitKey(0)

    print("Word Segmenting Done.")