import numpy as np
import cv2
import matplotlib.pyplot as plt

def SNR(x,y):
  n = np.sum(np.power(x,2))
  d = np.sum(np.power((x-y),2))
  snr = 10*np.log10(n/d)
  return snr

x = cv2.imread('city_orig.jpg', 0).astype(np.float64)
x_noisy = cv2.imread('city_noise.jpg', 0).astype(np.float64)
n = int(x.shape[0]/2)
m = int(x.shape[1]/2)
I_origin = np.zeros((n,m,4))
I_noisy = np.zeros((n,m,4))
k = 0
for i in range(2):
  for j in range(2):
    I_origin[:,:,k] = x[i*n:(i+1)*n,j*m:(j+1)*m]
    I_noisy[:,:,k] = x_noisy[i*n:(i+1)*n,j*m:(j+1)*m]
    k = k+1

snr_noisy = np.zeros((1,4))

for i in range(4):
  snr_noisy[0][i] = SNR(I_origin[:,:,i],I_noisy[:,:,i])

print(snr_noisy[0,:])

# mean filter
kernel_size = 5
mean_x = cv2.blur(x_noisy,(kernel_size, kernel_size))

plt.figure(figsize=(20,20))
plt.subplot(121)
plt.imshow(x_noisy,cmap='gray')
plt.title('Noisy Image')

plt.subplot(122)
plt.imshow(mean_x,cmap='gray')
plt.title('Output image result from Mean filter')


# Guassian Filter
kernel_size = 5
gaussian_x = cv2.GaussianBlur(x_noisy, (kernel_size, kernel_size),0)

plt.figure(figsize=(20,20))
plt.subplot(121)
plt.imshow(x_noisy,cmap='gray')
plt.title('Noisy Image')

plt.subplot(122)
plt.imshow(gaussian_x,cmap='gray')
plt.title('Output image result from Gaussian filter')

kernel_size = 5
x_noisy = x_noisy.astype('float32')
median_x = cv2.medianBlur(x_noisy, kernel_size)

plt.figure(figsize=(20,20))
plt.subplot(121)
plt.imshow(x_noisy,cmap='gray')
plt.title('Noisy Image')

plt.subplot(122)
plt.imshow(median_x,cmap='gray')
plt.title('Output image result from Median filter')

I_denoised_mean = np.zeros((n,m,4))
I_denoised_gaussian = np.zeros((n,m,4))
I_denoised_median = np.zeros((n,m,4))
k = 0
for i in range(2):
  for j in range(2):
    I_denoised_mean[:,:,k] = mean_x[i*n:(i+1)*n,j*m:(j+1)*m]
    I_denoised_gaussian[:,:,k] = gaussian_x[i*n:(i+1)*n,j*m:(j+1)*m]
    I_denoised_median[:,:,k] = median_x[i*n:(i+1)*n,j*m:(j+1)*m]
    k = k+1


snr_denoised_mean = np.zeros((1,4))
snr_denoised_gaussian = np.zeros((1,4))
snr_denoised_median = np.zeros((1,4))

for i in range(4):
  snr_denoised_mean[0][i] = SNR(I_origin[:,:,i],I_denoised_mean[:,:,i])
  snr_denoised_gaussian[0][i] = SNR(I_origin[:,:,i],I_denoised_gaussian[:,:,i])
  snr_denoised_median[0][i] = SNR(I_origin[:,:,i],I_denoised_median[:,:,i])

print(snr_denoised_mean[0,:])
print(snr_denoised_gaussian[0,:])
print(snr_denoised_median[0,:])


import pandas as pd
data = [snr_noisy[0,:],snr_denoised_mean[0,:], 
        snr_denoised_median[0,:],snr_denoised_gaussian[0,:]]

df = pd.DataFrame(data, columns=['First Part: Salt & paper noise','Second Part: Without noise', 'Third Part: Both types of Noise',
                                 'Forth Part: Gaussian noise'], index=['SNR befor apply filter',
                               'SNR after apply Mean filter',
                               'SNR after apply Median filter',
                              'SNR after apply Gaussian filter'])

display(df)