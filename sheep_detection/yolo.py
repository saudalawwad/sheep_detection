import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
from yolo_utils import infer_image, show_image



if __name__ == '__main__':


	# Get the labels
	labels = open("./yolov3-coco/coco-labels").read().strip().split('\n')

	# Intializing colors to represent each label uniquely

	colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
	

	# Load the weights and configutation to form the pretrained YOLOv3 model
	net = cv.dnn.readNetFromDarknet("./yolov3-coco/yolov3.cfg", "./yolov3-coco/yolov3 (2).weights")

	# Get the output layer names of the model
	layer_names = net.getLayerNames()
	
	layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        
	# If both image and video files are given then raise error
	

	# Do inference with given image
	

	if "sheep walking.mp4":
		# Read the video
		try:
			vid = cv.VideoCapture("sheep walking.mp4")
			height, width = None, None
			writer = None
		except:
			raise 'Video cannot be loaded!\n\
                               Please check the path provided!'

		finally:
			while True:
				grabbed, frame = vid.read()
				
			    # Checking if the complete video is read
				if not grabbed:

					break

				if width is None or height is None:
					height, width = frame.shape[:2]

				frame, _, _, _, _,list_of_plot = infer_image(net, layer_names, height, width, frame, colors, labels, 0)
 				
				
				print(list_of_plot)
				if writer is None:
					
					# Initialize the video writer
					fourcc = cv.VideoWriter_fourcc(*"MJPG")
					writer = cv.VideoWriter("./output.avi", fourcc, 30, 
						            (frame.shape[1], frame.shape[0]), True)


				writer.write(frame)

			print ("[INFO] Cleaning up...")
			writer.release()
			vid.release()


	
