import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

img = cv2.imread("q5.png")

fig = plt.figure(figsize=(20,20))
plt.subplot(1,3,1)
plt.imshow(img[:,:,2],cmap='gray')
#plt.imshow(cv2.cvtColor(img[:,:,0], cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Red dimension of image")

plt.subplot(1,3,2)
plt.imshow(img[:,:,1],cmap='gray')
plt.axis("off")
plt.title("Green dimension of image")

plt.subplot(1,3,3)
plt.imshow(img[:,:,0],cmap='gray')
plt.axis("off")
plt.title("Blue dimension of image")

mask = np.zeros_like(img[:,:,1],dtype=np.uint8)
for i in range(img.shape[0]):
  for j in range(img.shape[1]):
    if img[i,j,1]==255:
      mask[i,j] = 255

fig = plt.figure(figsize=(10,10))
plt.imshow(mask,cmap='gray')
plt.axis("off")
plt.title("Mask for clothes")
plt.savefig('q5res01.jpg')

def ero(img, k):
  m,n= img.shape
  SE= np.ones((k,k))
  constant= math.floor(k//2)
  imgErode= np.zeros((m,n),dtype=np.uint8)
  for i in range(constant, m-constant):
    for j in range(constant,n-constant):
      temp= img[i-constant:i+constant+1, j-constant:j+constant+1]
      product= temp*SE
      imgErode[i,j]= np.min(product)
  return imgErode

def dil(img, k):
  m,n= img.shape
  SE= np.ones((k,k))
  constant= math.floor(k//2)
  imgDilate= np.zeros((m,n),dtype=np.uint8)
  for i in range(constant, m-constant):
    for j in range(constant,n-constant):
      temp= img[i-constant:i+constant+1, j-constant:j+constant+1]
      product= temp*SE
      imgDilate[i,j]= np.max(product)
  return imgDilate

def closing(img, k):
  return ero(dil(img, k), k)

def opening(img, k):
  return dil(ero(img, k), k)

c_mask = closing(mask, 11)
o_mask = opening(c_mask, 7)


fig = plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.imshow(c_mask,cmap='gray')
plt.axis("off")
plt.title("Mask after Closing")

plt.subplot(1,2,2)
plt.imshow(o_mask,cmap='gray')
plt.axis("off")
plt.title("Mask after Opening")
plt.savefig('q5res02.jpg')

masked_img = img
for i in range(c_mask.shape[0]):
  for j in range(c_mask.shape[1]):
    if(c_mask[i,j]==255):
      masked_img[i,j,0] = 0
      masked_img[i,j,1] = 0
      masked_img[i,j,2] = 255

fig = plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Image after masking")
plt.savefig('q5res03.jpg')

