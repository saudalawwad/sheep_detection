#from tkinter import *
#from PIL import ImageTk,Image
from tkinter import filedialog
from tkhtmlview import HTMLLabel
import tkinter as tk, threading
from PIL import ImageTk,Image
import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
from yolo_utils import infer_image, show_image

import imageio
import matplotlib.pyplot as plt



def UploadAction(event=None):
    filename = filedialog.askopenfilename()
    
    
    
  
    
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
    

    if filename:
        # Read the video
        try:
            vid = cv.VideoCapture(filename)
            height, width = None, None
            writer = None
        except:
            raise 'Video cannot be loaded!\n\
                               Please check the path provided!'

        finally:
            list_of_plot = []
            
            while True :
                grabbed, frame = vid.read()
                
                # Checking if the complete video is read
                if not grabbed:

                    break

                if width is None or height is None:
                    height, width = frame.shape[:2]

                frame,count, _, _, _, _, = infer_image(net, layer_names, height, width, frame, colors, labels, 0)
                
                
                

                list_of_plot.append(count)
                if writer is None:
                    
                    # Initialize the video writer
                    fourcc = cv.VideoWriter_fourcc(*"MJPG")
                    writer = cv.VideoWriter("./output.avi", fourcc, 30, 
                                    (frame.shape[1], frame.shape[0]), True)


                writer.write(frame)
            writer.release()
            vid.release()
            tk.messagebox.showinfo("showinfo","Video Ready and it saved in the same directory with name output.avi")


            from os import startfile
            startfile("output.avi")
            
            x_axis = []
            x = 0.006

            for n in range(len(list_of_plot)):
                x_axis.append(x)
                x=x+0.006
            y = np.array(list_of_plot)
            x = np.array(x_axis)

            plt.plot(x,y,marker = 'o')
            plt.xlabel("frame")
            plt.ylabel("number of detection")
            plt.show()



            
            
            


  

root = tk.Tk(className = 'Python')
root.geometry("400x400")

image2 =Image.open('sheeps_img.jpg')
image1 = ImageTk.PhotoImage(image2)

image2 = image2.resize((1600,900))

root.geometry(f"{image2.size[0]}x{image2.size[1]}")
background_image= ImageTk.PhotoImage(image2)
background_label = tk.Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

button = tk.Button(root, text='Open MP4 file', command=UploadAction,activebackground = "#ffffff",activeforeground= "#5BB735",bg = "#66E432",highlightbackground = "red",
	bd =0 ,highlightcolor = "#5BB735",font = 24)
button.place(relx=0.5, rely=0.5, anchor="e")






root.mainloop()










