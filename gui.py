#!/usr/bin/env python
# coding: utf-8

# # Step 7: Building the Interface using Tkinter

# ## Pre-requisites if only running interface and not whole file

# In[1]:


import tkinter as tk
import PIL
import cv2
import csv
import numpy as np
from keras.models import load_model
from PIL import ImageTk
from tkinter import *
from tkinter import filedialog
from tqdm import tqdm
from skimage import exposure

# load the previously trained model
model = load_model('final_model.h5')

labels_dict = None
with open('mapSignnamesToClass.csv', mode='r') as infile:
    reader = csv.reader(infile)
    next(reader, None)
    labels_dict = {int(rows[0]): rows[1] for rows in reader}

# Function that applies normalization and local contrast enhancement


def normalize(image_data):
    '''Contrast Limited Adaptive Histogram Equalization (CLAHE). In addition to regular normalization, 
    this function provides local contrast enhancement -- i.e., details of the image can be 
    enhanced even in regions that are darker or lighter than most of the image.
    http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_adapthist
    '''

    norm = np.array([exposure.equalize_adapthist(image, clip_limit=0.1)
                    for image in tqdm(image_data)])
    return norm


# ## Interface

# In[2]:


# initialise GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Traffic Sign Recognition System usig CNN')
top.configure(background='#3F4E4F')

# Heading
heading = Label(top, text="TRAFFIC SIGN CLASSIFICATION",
                pady=20, font=('MS Sans Serif', 22, 'bold'))
heading.configure(background='#3F4E4F', foreground='#DCD7C9')
heading.pack()

# Frame for button section
btn_frame = Frame(top, background='#3F4E4F')
btn_frame.pack()
btn_frame.place(anchor='e', relx=0.95, rely=0.5)

classify_btn_frame = Frame(btn_frame, background='#3F4E4F')
classify_btn_frame.pack(side=TOP)

# Frame for Image section
image_frame = Frame(top, width=400, height=400, background='#3F4E4F')
image_frame.pack()
image_frame.place(anchor='w', relx=0.15, rely=0.5)

sign_image = Label(image_frame, background='#3F4E4F')
sign_image.pack(side=TOP, pady=10)

label = Label(image_frame, background='#3F4E4F',
              foreground='#DCD7C9', font=('MS Sans Serif', 17))
label.pack(side=BOTTOM)

# Predict the class


def classify(file_path):
    global label_packed
    image = cv2.imread(file_path)
    image = PIL.Image.fromarray(image, 'RGB')
    image = image.resize((30, 30))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    image = image/255
    image = normalize(image)
    pred = model.predict(image)
    result = pred.argmax()
    sign = labels_dict[result]
    print(sign)
    label.configure(text=sign)

# Show classify button when image is uploaded


def show_classify_button(file_path):
    classify_btn.configure(command=lambda: classify(file_path))
    classify_btn.pack(side=TOP)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = PIL.Image.open(file_path)
        uploaded = uploaded.resize((400, 300))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass


upload_btn = Button(btn_frame, text="Browse Image",
                    command=upload_image, padx=15, pady=10)
upload_btn.configure(background='#DCD7C9',
                     foreground='#2C3639', font=('Courier New', 12, 'bold'))
upload_btn.pack(side=BOTTOM, pady=20)

classify_btn = Button(classify_btn_frame,
                      text="Classify Sign", padx=11, pady=8)
classify_btn.configure(background='#DCD7C9',
                       foreground='#2C3639', font=('Courier New', 12, 'bold'))

top.mainloop()
