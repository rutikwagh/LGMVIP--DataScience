#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install opencv-python


# ### The only library we need for converting an image into a pencil sketch with Python is an OpenCV library in Python. It can be used by using the pip command; pip install opencv-python. But it is not imported by the same name. Let’s import it to get started with the task:

# In[ ]:


#import the library
import cv2


# I will not display the image at every step, if you want to display the image at every step to see the changes in the image then you need to use two commands; cv2.imshow(“Title You want to give”, Image) and then simply write cv2.waitKey(0). This will display the image.
# 
# Now the next thing to do is to read the image:

# In[3]:


#read in the image
img = cv2.imread("opencv.jpg")


# Now after reading the image, we will create a new image by converting the original image to greyscale:

# In[4]:


#convert the image to gray scale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Now the next step is to invert the new grayscale image:

# In[5]:


#invert the image
inverted_gray_image = 255 - gray_image


# Now the next step in the process is to blur the image by using the Gaussian Function in OpenCV:

# In[6]:


#blur the image by gaussian blur
blurred_image = cv2.GaussianBlur(inverted_gray_image, (21, 21), 0)


# Then the final step is to invert the blurred image, then we can easily convert the image into a pencil sketch:

# In[7]:


#invert the blurred image
inverted_blurred_image = 255 - blurred_image


# And finally, if you want to have a look at both the original image and the pencil sketch then you can use the following commands:

# In[8]:


#create the pencil sketch image
pencil_sketch_image = cv2.divide(gray_image, inverted_blurred_image, scale = 256.0)


# In[ ]:


#show the image
cv2.imshow('Original Image', img)

cv2.imshow('New Image', pencil_sketch_image)

cv2.waitKey(0)

