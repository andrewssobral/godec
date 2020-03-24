import argparse
import time

from godec import godec
from utils import play_2d_results, play_2d_video

__doc__ = """
Decompose an input file into its low-rank and sparse components using the GoDec algorithm.
"""


def process_video_file(input_file, debug=False):
    # Import OpenCV
    from numpy import array, column_stack
    try:
        import cv2 as cv
    except ImportError:
        raise ImportError('OpenCV is requires to read video files')

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv.VideoCapture(input_file)

    # Check if video was opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    # Read until video is completed
    M = None
    height = None
    width = None
    i = 0
    print("Press 'q' to stop...")
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            height, width = frame.shape

            # Stack frames as column vectors
            F = frame.T.reshape(-1)

            if i == 0:
                M = array([F]).T
            else:
                M = column_stack((M, F))

            if debug:
                # Display the resulting frame
                cv.imshow('Frame', frame)

                # Press Q on keyboard to exit
                if cv.waitKey(25) & 0xFF == ord('q'):
                    break

            i = i + 1
            # print(i)

        # Break the loop
        else:
            break

    # Release the video capture object
    cap.release()

    # Closes all the frames
    if debug:
        cv.destroyAllWindows()

    assert M is not None
    assert width is not None
    assert height is not None

    print("Number of frames: ", i, " with size ", (width, height))
    print("Processing M with shape ", M.shape)
    run_godec(M, width, height, debug)


def process_matlab_file(input_file, debug=False):
    import scipy.io as sio
    # Load data
    mat = sio.loadmat(input_file)
    M, height, width = mat['M'], int(mat['m']), int(mat['n'])
    if debug:
        play_2d_video(M, width, height)
    run_godec(M, width, height, debug)


def run_godec(M, width, height, debug=False):
    # Run GoDec
    t = time.time()
    L, S, LS, _ = godec(M)
    elapsed = time.time() - t
    print(elapsed, "sec elapsed")
    # Plot results
    if debug:
        play_2d_results(M, LS, L, S, width, height)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'input_file',
        nargs='?',
        type=str,
        const=1,
        default='dataset/demo.avi',
        help="The input file should be a video file (avi, mpg, etc.) or a MATLAB-style file (.mat)"
    )
    parser.add_argument(
        'debug',
        nargs='?',
        type=bool,
        const=1,
        default=False,
        help="Enable visual debug"
    )
    args = parser.parse_args()

    input_file = args.input_file
    if input_file.endswith('.mat'):
        process_matlab_file(input_file, args.debug)
    else:
        process_video_file(input_file, args.debug)


if __name__ == "__main__":
    main()
