import argparse
import sys
import cv2


def convert(input, output, start, double):
    video = cv2.VideoCapture(input)
    fps = video.get(cv2.CAP_PROP_FPS)
    print('fps', fps)

    # navigate to start time
    if start > 0:
        video.set(cv2.CAP_PROP_POS_FRAMES, int(fps * start));

    success, image = video.read()
    count = 0
    while success:
      cv2.imwrite(output + "frame%d.jpg" % count, image)
      return
      success,image = vidcap.read()
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
