import io

# Import libraries
import numpy as np
import cv2
import os
import tensorflow as tf


class Video_Inference:

  def __init__(self,params = None):
    if params != None:
      video_params = params.inference.video
      
      self._output_size = params.input.image_size
      if self._output_size is None:
        self._output_size = video_params.default_size
        print("Assuming default size: {}}".format(self._output_size))

  def _check_valid_files(self,input_file, output_file):
    """
      Checks that input_file exists, output_file doesn't exist
        and that they are different from each other
        
      Args:
        input_file: the video input file path
        output_file: the video output file path
    """
    input_exists = os.path.exists(input_file)
    if not input_exists:
      raise ValueError("Input File: '{}' does not exist!".format(input_file))
    
    output_exists = os.path.exists(output_file)
    if output_exists:
      raise ValueError("Output file: '{}' exist!".format(output_file))
      
    if(input_file == output_file):
      raise ValueError("Same input and output file path: '{}'".format(input_file))
    

  def _format_frames(self,frame):
    """
      Pad and resize an image from a video.

      Args:
        frame: Image that needs to resized and padded. 
        output_size: Pixel size of the output frame image.

      Return:
        Formatted frame with padding of specified output size.
    """
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, self._output_size[0], self._output_size[1])
    frame = tf.image.convert_image_dtype(frame, tf.uint8)
    return frame


  def process_save_video(self,
                         input_file,
                         output_file,
                         model = None,
                         codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')):
    """
      Loads and performs inference on a video.
      
      Args:
        input_file: input video path
        output_file: output video file
        model: Modeul used for frame inference
        codec: Codec used for encoding video for saving
    """
    self._check_valid_files(input_file,output_file)
    
    result = self.video_inference(model,input_file)
    
    fps,_,_ = self._get_video_info(input_file)
    (height,width,_) = (np.shape(result[0]))
    out = cv2.VideoWriter(output_file, fourcc=codec, fps=fps, frameSize=(width,height))
    for i in range(len(result)):
      out.write(result[i].astype(np.uint8))
    out.release()
    
    
    
  def _get_video_info(self, input_file):
    """
      Returns video metadata
      
      Args:
        input_file: video file path
      Return:
        fps, video dimensions (height, width), frame count
    """
    src = cv2.VideoCapture(input_file)
    fps = src.get(cv2.CAP_PROP_FPS)
    height = src.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = src.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_count = src.get(cv2.CAP_PROP_FRAME_COUNT)
    src.release()
    return int(fps), (int(width),int(height)), int(frame_count)
    


  def video_inference(self,model,input_file):
    """
      Creates frames from each video file present for each category.

      Args:
        model: Model used for inference on frame
        input_file: File path to the video.
      Return:
        An NumPy array of frames in the shape of (n_frames, height, width, channels).
    """
    # Read each video frame by frame
    
    src = cv2.VideoCapture(str(input_file))  

    video_length = (int)(src.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if video_length == 0:
      raise ValueError("Found video with '{}' frames.".format(video_length))
    
    result = []
    print("parsing {} frames...".format(video_length))
    if model is None:
      print("Model is None, not running prediction with model!")
      
    # ret is a boolean indicating whether read was successful
    # frame is the image itself
    
    for count in range(video_length):
      print("parsing {}/{}".format(count,video_length),end = "\r")
      success_read, frame = src.read()
      if success_read:
        if model != None:
          # run inference on model
          inp = tf.convert_to_tensor(frame)
          frame = model.predict(inp)
          frame = frame.numpy()
      else:
        frame = np.zeros_like(result[0])
        
      frame = self._format_frames(frame)
      result.append(frame)
    src.release()
    result = np.array(result)
    
    print("output shape: {}".format(result.shape))
    return result

#   def test_video_inference(self):
#     input_path = "/home/abuynits/projects/test2.MOV"
#     output_path = "/home/abuynits/projects/output_test.mp4"
#     self.process_save_video(
#       input_path,
#       output_path,
#     )
    
    
# v = Video_Inference()
# v.test_video_inference()