import io

# Import libraries
import numpy as np
import cv2
import tensorflow as tf

def format_frames(frame, output_size):
  """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded. 
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
  """
  frame = tf.image.convert_image_dtype(frame, tf.float32)
  frame = tf.image.resize_with_pad(frame, *output_size)
  return frame




def load_video_frames(video_path, 
                      output_size = (224,224), 
                      frame_step = 15):
  """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
  """
  # Read each video frame by frame
  result = []
  src = cv2.VideoCapture(str(video_path))  

  video_length = (int)(src.get(cv2.CAP_PROP_FRAME_COUNT))
  print("\n\n\n\n\n")
  print(video_length)
  print("\n\n\n\n\n")
  # ret is a boolean indicating whether read was successful
  # frame is the image itself
  ret, frame = src.read()
  result.append(format_frames(frame, output_size))
  for _ in range(video_length):
    success_read, frame = src.read()
    print(success_read)
    print(frame)
    if success_read:
      frame = format_frames(frame,output_size)
    else:
      frame = np.zeros_like(result[0])
    result.append(frame)

  src.release()
  result = np.array(result)[..., [2, 1, 0]]

  return result

vid_path = '/home/abuynits/projects/test.MOV'
res = load_video_frames(vid_path)