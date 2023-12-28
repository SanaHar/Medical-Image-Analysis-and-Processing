import cv2
from matplotlib import pyplot as plt
import numpy as np

def derivative(I, mode):
  #coloured_image = cv2.imread(image_path)
  #grey_image = cv2.cvtColor(coloured_image, cv2.COLOR_BGR2GRAY)
  if mode=='centeral':
    K = np.array([[-1,0,1]])
  elif mode=='forward':
    K = np.array([[0,-1,1]])
  elif mode=='backward':
    K = np.array([[-1,1,0]])
    
  Fx = np.zeros_like(I)
  Fy = np.zeros_like(I)
  
  # Add zero padding to the input image
  I_x = np.zeros((I.shape[0], I.shape[1]+2))
  I_x[:, 1:-1] = I
  
  I_y = np.zeros((I.shape[0] + 2, I.shape[1]))
  I_y[1:-1, :] = I

  # Loop over every pixel of the image
  for x in range(I.shape[0]):
    for y in range(I.shape[1]):
      # element-wise multiplication of the kernel and the image
      Fx[x, y] = (K * I_x[x, y: y+3]).sum()
      Fy[x, y] = (K * I_y[x:x+3, y]).sum()
  
  dI = np.sqrt(Fx**2 + Fy**2) 
  return abs(Fx), abs(Fy), dI
  
I = cv2.imread('q1.png', 0).astype(np.float64)

Fx1, Fy1, dI1 = derivative(I, 'centeral')

Fx2, Fy2, dI2 = derivative(I, 'forward')

Fx3, Fy3, dI3 = derivative(I, 'backward')

# create figure
fig = plt.figure(figsize=(20, 20))


fig.add_subplot(4,3,2)
plt.imshow(I, cmap='gray')
plt.axis('off')
plt.title("Original Image")
  

fig.add_subplot(4,3,4)
plt.imshow(Fx1, cmap='gray')
plt.axis('off')
plt.title("Centeral difference: Fx")

fig.add_subplot(4,3,5)
plt.imshow(Fy1, cmap='gray')
plt.axis('off')
plt.title("Centeral difference: Fy")

fig.add_subplot(4,3,6)
plt.imshow(dI1, cmap='gray')
plt.axis('off')
plt.title("Centeral difference: sqrt(Fy^2+Fx^2)")

fig.add_subplot(4,3,7)
plt.imshow(Fx2, cmap='gray')
plt.axis('off')
plt.title("Forward difference: Fx")

fig.add_subplot(4,3,8)
plt.imshow(Fy2, cmap='gray')
plt.axis('off')
plt.title("Forward difference: Fy")

fig.add_subplot(4,3,9)
plt.imshow(dI2, cmap='gray')
plt.axis('off')
plt.title("Forward difference: sqrt(Fy^2+Fx^2)")


fig.add_subplot(4,3,10)
plt.imshow(Fx3, cmap='gray')
plt.axis('off')
plt.title("Backward difference: Fx")

fig.add_subplot(4,3,11)
plt.imshow(Fy3, cmap='gray')
plt.axis('off')
plt.title("Backward difference: Fy")

fig.add_subplot(4,3,12)
plt.imshow(dI3, cmap='gray')
plt.axis('off')
plt.title("Backward difference: sqrt(Fy^2+Fx^2)")