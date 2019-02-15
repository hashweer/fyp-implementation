import cv2
import numpy as np
import os
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt


def resize_character(image):

    height, width = image.shape
    resize_height, resize_width = 50, 50
    left, right, top, bottom = 0, 0, 0, 0

    if width < resize_height:
        difference = resize_height - width

        left = difference // 2
        right = difference // 2

        if difference % 2 == 1:
            left = left + 1

    if height < resize_height:
        difference = resize_height - height

        top = difference // 2
        bottom = difference // 2

        if difference % 2 == 1:
            top = top + 1

    image = cv2.copyMakeBorder(image, top=top, bottom=bottom, left=left, right=right, borderType=cv2.BORDER_CONSTANT,
                               value=[255, 255, 255])

    return image


def get_re_segment_count(img):
    img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]  # ensure binary
    img = 255 - img

    kernel = np.ones((1, 1), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)

    img[img == 255] = 1

    array = np.array(img)
    structure = np.ones((3, 3), dtype=np.int)
    _, no_of_components = label(array, structure)

    return no_of_components


def re_segment(img):
    img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]  # ensure binary
    img = 255 - img

    kernel = np.ones((1, 1), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)

    img[img == 255] = 1

    array = np.array(img)
    structure = np.ones((3, 3), dtype=np.int)
    labeled, no_of_components = label(array, structure)

    segments = dict()

    for y in range(0, no_of_components):
        labeled_copy = labeled.copy()
        labeled_copy[labeled_copy != y + 1] = 0
        labeled_copy[labeled_copy == y + 1] = 255
        # labeled_copy = 255 - labeled_copy

        pixel_sum = np.sum(labeled_copy, axis=0).tolist()
        pixel_sum[0] = 0
        pixel_sum[len(pixel_sum) - 1] = 0
        cords = []

        for x in range(len(pixel_sum) - 1):
            current_value = pixel_sum[x]
            next_value = pixel_sum[x + 1]

            if current_value == 0 and next_value == 0:
                continue
            elif current_value > 0 and next_value > 0:
                continue
            elif current_value == 0 and next_value > 0:
                cords.append(x)
            elif current_value > 0 and next_value == 0:
                cords.append(x + 1)

        H, W = labeled_copy.shape[:2]
        roi = labeled_copy[0:H, cords[0]:cords[1]]
        roi = cv2.dilate(roi.astype(np.float32), kernel, iterations=1)
        roi = 255 - roi

        resized = resize_character(roi)
        segments[cords[0]] = resized

    sorted_segments = []
    segments = sorted(segments.items())
    for key, value in segments:
        sorted_segments.append(value)

    return sorted_segments


def character_segment(base_path, character_segments_path, image_name):
    print("Character Segmenting Started...")

    base_image_name_array = image_name.split('.')
    path = base_path + character_segments_path + base_image_name_array[0] + '/'

    line_list = []
    for (dir_path, dir_names, file_names) in os.walk(path):
        line_list.extend(file_names)
        break

    sorted_word_list = []

    for i in range(len(line_list)):
        image_name = line_list[i]
        image_name_array = image_name.split('.')
        new_image_name = image_name_array[0].replace('_', '.')
        sorted_word_list.append(float(new_image_name))

    sorted_word_list.sort()

    character_segment_base_path = 'step_11_character_segment/'
    character_segmented_path = character_segment_base_path + 'segmented/' + base_image_name_array[0]

    if not os.path.exists(base_path + character_segmented_path):
        os.makedirs(base_path + character_segmented_path)

    # character_count = 0

    for i in range(len(sorted_word_list)):

        reset_image_name = str(sorted_word_list[i]).replace('.', '_')

        line_segment_original = cv2.imread(path + reset_image_name + '.' + base_image_name_array[1])
        line_segment = cv2.imread(path + reset_image_name + '.' + base_image_name_array[1], 0)

        if line_segment is not None:
            line_segment = 255 - line_segment
            line_segment_sum = np.sum(line_segment, axis=0).tolist()

            line_segment_sum[0] = 0
            line_segment_sum[len(line_segment_sum) - 1] = 0

            # ignore counts less than 800
            for x in range(len(line_segment_sum)):
                if line_segment_sum[x] < 800:
                    line_segment_sum[x] = 0

            # print(line_segment_sum)

            # identify rising points and falling points
            cords = []
            for x in range(len(line_segment_sum) - 1):
                current_value = line_segment_sum[x]
                next_value = line_segment_sum[x + 1]

                if current_value == 0 and next_value == 0:
                    continue
                elif current_value > 0 and next_value > 0:
                    continue
                elif current_value == 0 and next_value > 0:
                    cords.append(x+1)
                elif current_value > 0 and next_value == 0:
                    cords.append(x)

            # print(cords)

            H, W = line_segment.shape[:2]

            differences = []

            # get line differences
            for x in range(len(cords) - 1):
                current_value = cords[x]
                next_value = cords[x + 1]

                difference = next_value - current_value

                # filter values
                if 1 < difference < 100:
                    differences.append(difference)
                else:
                    differences.append(0)

            # print(differences)

            # detect pairs
            pairs = []
            for x in range(len(differences)):

                pair = []
                if 1 < differences[x] < 80:
                    pair.append(cords[x])
                    pair.append(cords[x + 1])
                    pairs.append(pair)

            line_segment = 255 - line_segment

            # print(pairs)

            character_cords = [0]

            for x in range(len(pairs)):

                pair = pairs[x]
                roi = line_segment[0:H, pair[0]:pair[1]]

                # count black pixels
                black_pixel_count = np.sum(roi == 0)

                if black_pixel_count < 15:
                    character_cords.append(pair[0])
                    character_cords.append(pair[1])

            character_cords.append(W)

            # print(character_cords)

            # arrange new pairs
            arranged_pairs = []
            for x in range(0, len(character_cords), 2):
                if character_cords[x] != character_cords[x + 1]:
                    pair = [character_cords[x], character_cords[x + 1]]
                    arranged_pairs.append(pair)
            # print(arranged_pairs)
            # draw new lines

            character_count = 0
            for x in range(len(arranged_pairs)):

                pair = arranged_pairs[x]
                roi = line_segment[0:H, pair[0]:pair[1]]

                cv2.line(line_segment_original, (pair[0], 0), (pair[0], H), (0, 0, 255), 1)
                cv2.line(line_segment_original, (pair[1], 0), (pair[1], H), (255, 0, 0), 1)

                roi_h, roi_w = roi.shape[:2]

                if roi_w > 35:
                    if get_re_segment_count(roi) > 1:
                        re_segmented_items = re_segment(roi)

                        for item in re_segmented_items:
                            character_count += 1
                            cv2.imwrite(os.path.join(base_path + character_segmented_path,
                                                     str(i+1) + '_' + str(character_count) + '.' + base_image_name_array[1]), item)
                    else:
                        roi = resize_character(roi)
                        character_count += 1
                        cv2.imwrite(os.path.join(base_path + character_segmented_path,
                                                 str(i+1) + '_' + str(character_count) + '.' + base_image_name_array[1]), roi)
                else:
                    roi = resize_character(roi)
                    character_count += 1
                    cv2.imwrite(os.path.join(base_path + character_segmented_path,
                                         str(i+1) + '_' + str(character_count) + '.' + base_image_name_array[1]), roi)



            # print("----------------------------------")

            # cv2.imwrite(os.path.join(base_path + character_segmented_path + (base_image_name_array[0] + str(i) + '.' + base_image_name_array[1])), line_segment_original)
            # cv2.imshow('segmented', line_segment_original)
            # plt.plot(line_segment_sum)
            # plt.show()
            # cv2.waitKey(0)

    print("Character Segmenting Done.")
