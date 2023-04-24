import os
import cv2

# Set input and output directories
# input_dir = '/path/to/input/videos'
# output_dir = '/path/to/output/videos'

input_dir = 'D:/CV/yolov7-main/resize_vid_1'
output_dir = 'D:/CV/yolov7-main/resize_vid_2'

# Set output video properties
fps = 20
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Loop over input videos
for filename in os.listdir(input_dir):
    if filename.endswith('.mp4'):
        # Open video file
        input_file = os.path.join(input_dir, filename)
        cap = cv2.VideoCapture(input_file)

        # Get input video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Set output video dimensions and filename
        output_file = os.path.join(output_dir, filename)
        out = cv2.VideoWriter(output_file, fourcc, fps, (1920, 1080))

        # Loop over video frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame
            frame = cv2.resize(frame, (1920, 1080))

            # Write frame to output video
            out.write(frame)

        # Release video capture and writer objects
        cap.release()
        out.release()

print('Done')