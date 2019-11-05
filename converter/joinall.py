import argparse
import cv2
import os
import numpy as np
import glob
import math
import sintel_io


def join_color(inputs, output, width):
    FULL_SIZE = 4096

    COLUMNS_PER_ROW = int(math.floor(FULL_SIZE / width))

    # get size from first frame
    frame = cv2.imread(inputs[0], cv2.IMREAD_COLOR)
    height, orig_width, _ = frame.shape

    print("max count", int(math.floor(FULL_SIZE / height)) * COLUMNS_PER_ROW - 1)
    print("""h w
    color: {{
        source: "{}",
        width: {} / {},
        height: {} / {},
        per_row: {}
    }}
    """.format(output, width, FULL_SIZE, height, FULL_SIZE, COLUMNS_PER_ROW))

    # prepare the result
    result = np.zeros((FULL_SIZE, FULL_SIZE, 3), np.uint8)

    for i, input in enumerate(inputs):
        print('frame', i)
        frame = cv2.imread(input, cv2.IMREAD_COLOR)
        dx = (orig_width - width) // 2
        frame = frame[:, dx:dx+width]

        # place image
        row, col = divmod(i, COLUMNS_PER_ROW)
        y = row * height
        x = col * width
        result[y:y+height, x:x+width] = frame

    cv2.imwrite(output, result)


def join_depth(inputs, output, width):
    FULL_SIZE = 4096

    COLUMNS_PER_ROW = int(math.floor(FULL_SIZE / width))

    # get size from first frame
    frame = sintel_io.depth_read(inputs[0])
    height, orig_width = frame.shape

    all = []
    for i, input in enumerate(inputs):
        depth = sintel_io.depth_read(input)
        dx = (orig_width - width) // 2
        depth = depth[:, dx:dx+width]
        all.append(depth)

    # calculate params
    comb = np.concatenate(all, axis=1)
    mn = comb.min()
    mx = comb.max()
    img = ((comb - mn) * 255 / (mx - mn)).astype(np.uint8)
    ret, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    front_point = mn + (mx - mn) * (ret / 255)
    FRONT_COLOR = 145  # from apple

    print("""h w
    color: {{
        source: "{}",
        width: {} / {},
        height: {} / {},
        per_row: {}
    }}
    """.format(output, width, FULL_SIZE, height, FULL_SIZE, COLUMNS_PER_ROW))

    # prepare the result
    result = np.zeros((FULL_SIZE, FULL_SIZE), np.uint8)

    for i, frame in enumerate(all):
        print('frame', i)

        front_mask = (frame < front_point)

        frame[front_mask] = (
            FRONT_COLOR +
            (front_point - frame[front_mask]) /
            (front_point - mn) * (255 - FRONT_COLOR))

        frame[~front_mask] = (
            (mx - frame[~front_mask]) /
            (mx - front_point) * FRONT_COLOR)

        # place image
        row, col = divmod(i, COLUMNS_PER_ROW)
        y = row * height
        x = col * width
        result[y:y+height, x:x+width] = frame.astype(np.uint8)

    cv2.imwrite(output, result)


COLOR_SOURCE = '/home/denys/projects/jr/depth/MPI-Sintel-training_images/training/final/'
DEPTH_SOURCE = '/home/denys/projects/jr/depth/MPI-Sintel-depth-training-20150305/training/depth/'

COLOR_DESTINATION = '/home/denys/prj-shared/personal/25d-gif/preview/'
DEPTH_DESCTINATION = '/home/denys/prj-shared/personal/25d-gif/preview/'


for filename in glob.glob(COLOR_SOURCE + '*'):
    name = filename.split('/')[-1]
    print(name)
    continue
    color_s = COLOR_SOURCE + name + '/*'
    inputs = sorted(glob.glob(color_s))
    join_color(inputs, COLOR_DESTINATION + name + '_color.jpg', 480)

    depth_s = DEPTH_SOURCE + name + '/*'
    inputs = sorted(glob.glob(depth_s))
    join_depth(inputs, DEPTH_DESCTINATION + name + '_depth.png', 480)
