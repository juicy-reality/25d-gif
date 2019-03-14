import argparse
import cv2
import os
import numpy as np


def convert(input, output, start, max_count, skip):
    video = cv2.VideoCapture(input)

    # create folders
    os.makedirs(output + '/color/', exist_ok=True)
    os.makedirs(output + '/depth/', exist_ok=True)

    # navigate to start time
    fps = video.get(cv2.CAP_PROP_FPS)
    if start > 0:
        video.set(cv2.CAP_PROP_POS_FRAMES, int(fps * start));
    
    # read info and first frame    
    
    success, frame = video.read()
    count = 0
    
    print('fps', fps)
    while success:
        print('Processing frame: ', count)

        # resize frame
        height, width, _ = frame.shape

        # prepare for depth
        for_depth = cv2.resize(frame, (672, 192))
        cv2.imwrite(output + '/depth/' + output + "%03d.png" % count, for_depth)

        # save left
        c_width = int(width / 2)
        left = np.copy(frame[:, :c_width])        
        cv2.imwrite(output + '/color/' + output + "%03d.jpg" % count, left)

        # skip frames
        for i in range(skip):
            success, frame = video.read()
        
        # next frames
        success, frame = video.read()
        count += 1
        if count >= max_count:
            return


if __name__ == '__main__':
    desc = """frame extractor"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=desc)

    parser.add_argument('input', type=str, help='input video')

    parser.add_argument('output', type=str, help='output image file')

    parser.add_argument('--start', type=float, default=0, help='start in seconds')

    parser.add_argument('--count', type=int, default=1, help='max frames')

    parser.add_argument('--skip', type=int, default=0, help='skip frames')

    args = parser.parse_args()
    convert(args.input, args.output, args.start, args.count, args.skip)
