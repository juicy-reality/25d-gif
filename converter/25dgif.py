import argparse
import sys
import cv2
import numpy as np


def convert(input, output, start, double):
    video = cv2.VideoCapture(input)
    fps = video.get(cv2.CAP_PROP_FPS)
    print('fps', fps)

    # navigate to start time
    if start > 0:
        video.set(cv2.CAP_PROP_POS_FRAMES, int(fps * start));

    success, frame = video.read()
    count = 0
    while success:
        # resize frame
        height, width, _ = frame.shape
        if double:
            width = width * 2
        new_h, new_w = 512, int(width * 512 / height)
        frame = cv2.resize(frame, (new_w, new_h))

        # views
        c_width = int(new_w / 4)
        left = frame[:, c_width - 256:c_width + 256]
        right = frame[:, c_width * 3 - 256:c_width * 3 + 256]
        cv2.imwrite(output + "left%d.jpg" % count, left)
        cv2.imwrite(output + "right%d.jpg" % count, right)

        # depth map
        window_size = 3                     # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
 
        left_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=160,             # max_disp has to be dividable by 16 f. E. HH 192, 256
            blockSize=5,
            P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
            P2=32 * 3 * window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
  
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        # FILTER Parameters
        lmbda = 80000
        sigma = 1.2
        visual_multiplier = 1.0
         
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)
         
        print('computing disparity...')
        displ = left_matcher.compute(left, right)  # .astype(np.float32)/16
        dispr = right_matcher.compute(right, left)  # .astype(np.float32)/16
        displ = np.int16(displ)
        dispr = np.int16(dispr)
        filteredImg = wls_filter.filter(displ, left, None, dispr)  # important to put "imgL" here!!!
         
        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
        filteredImg = np.uint8(filteredImg)
        cv2.imwrite(output + "depth%d.jpg" % count, filteredImg)

        print(height, width)
        print(new_h, new_w)
        cv2.imwrite(output + "frame%d.jpg" % count, frame)
        return
        success, frame = video.read()
        print('Read a new frame: ', success)
        count += 1

if __name__ == '__main__':
    desc = """2,5d gif converter"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=desc)

    parser.add_argument('input', type=str, help='input video')

    parser.add_argument('output', type=str, help='output image file')

    parser.add_argument('--start', type=float, default=0, help='start in seconds')

    parser.add_argument('--double', help='scale width x2', action="store_true")

    args = parser.parse_args()
    convert(args.input, args.output, args.start, args.double)
