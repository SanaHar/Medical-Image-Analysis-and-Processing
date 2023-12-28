import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('q3_1.JPG',0)

# display the image
fig = plt.figure(figsize=(6, 8))
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title('Original Image')

bins = np.zeros(256,np.int32)

for i in range(img.shape[0]):
  for j in range(img.shape[1]):
    intensity = img[i][j]
    bins[intensity] +=1


fig = plt.figure(figsize=(20, 5))
plt.stem(np.arange(256),bins)
plt.grid('on')
plt.title("Histogram intensity of Original Image")
plt.xlabel("grayscale value")
plt.ylabel("pixel count")
plt.savefig("q3res1.jpg")

def get_histogram(img):
  #calculate the normalized histogram of an image
  h, w = img.shape
  hist = [0.0] * 256
  for i in range(h):
    for j in range(w):
      hist[img[i, j]]+=1
  return np.array(hist)/(h*w)

def get_cumulative_sums(hist):
  #find the cumulative sum of a numpy array
  return [sum(hist[:i+1]) for i in range(len(hist))]

def normalize_histogram(img):

  hist = get_histogram(img)
  cdf = np.array(get_cumulative_sums(hist))
  sk = np.uint8(255 * cdf)
  h, w = img.shape
  img_new = np.zeros_like(img)
  for i in range(0, h):
    for j in range(0, w):
      img_new[i, j] = sk[img[i, j]]

  hist_new = get_histogram(img_new)
  return img_new, hist_new

img_normalized, hist_normalized = normalize_histogram(img)

fig = plt.figure(figsize=(20, 5))
plt.stem(hist_normalized)
plt.grid('on')
plt.title("Normalized Histogram")
plt.savefig('q3res3.jpg')

fig = plt.figure(figsize=(20, 5))
plt.imshow(img_normalized,cmap = 'gray')
plt.axis('off')
plt.title("Normalized Image")
plt.savefig('q3res2.jpg')


def get_cumulative_sums(hist):
  #find the cumulative sum of a numpy array
  return [sum(hist[:i+1]) for i in range(len(hist))]

def hist_matching(original_img, target_img):
  hist = get_histogram(original_img)
  c = np.array(get_cumulative_sums(hist))

  hist = get_histogram(target_img)
  target_c = np.array(get_cumulative_sums(hist))
  
  b = np.interp(c, target_c, np.arange(256))
  
  pix_repl = {i:b[i] for i in range(256)} 
  mp = np.arange(0,256)
  for (k, v) in pix_repl.items():
    mp[k] = v
  s = original_img.shape
  original_img = np.reshape(mp[original_img.ravel()], original_img.shape)
  original_img = np.reshape(original_img, s)
  return original_img


def get_histogramOfRGB(img):
  #calculate the normalized histogram of an image
  hist = []
  for i in range(3):
    hist.append(get_histogram(img[:,:,i]))

  return hist


original_img = cv2.imread('q3_2.jpg')
target_img = cv2.imread('q3_3.jpg')

fig = plt.figure(figsize=(20, 5))
plt.subplot(121)
plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Original Image')

plt.subplot(122)
plt.imshow(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Target Image')


original_hist = get_histogramOfRGB(original_img)
fig = plt.figure(figsize=(20, 5))
plt.subplot(121)
plt.plot(original_hist[0],'r')
plt.plot(original_hist[1],'g')
plt.plot(original_hist[2],'b')
plt.title('Histogram of Original Image')
plt.grid('on')
plt.legend(['red','blue','green'])
plt.savefig('oq3res1.jpg')

target_hist = get_histogramOfRGB(target_img)
plt.subplot(122)
plt.plot(target_hist[0],'r')
plt.plot(target_hist[1],'g')
plt.plot(target_hist[2],'b')
plt.title('Histogram of Target Image')
plt.grid('on')
plt.legend(['red','blue','green'])
plt.savefig('oq3res2.jpg')


spe_img = np.zeros_like(original_img)
for i in range(3):
  spe_img[:,:,i]  = hist_matching(original_img[:,:,i], target_img[:,:,i])

fig = plt.figure(figsize=(20, 5))
plt.imshow(cv2.cvtColor(spe_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Final Image after histogram specification')
plt.savefig('oq3res3.jpg')

spe_hist = get_histogramOfRGB(spe_img)

fig = plt.figure(figsize=(20, 5))
plt.plot(spe_hist[0],'r')
plt.plot(spe_hist[1],'g')
plt.plot(spe_hist[2],'b')
plt.title('Histogram of specified Image')
plt.grid('on')
plt.legend(['red','blue','green'])
plt.savefig('oq3res4.jpg')

fig = plt.figure(figsize=(20, 5))
plt.plot((spe_hist[0]+spe_hist[1]+spe_hist[2])/3)
plt.plot((original_hist[0]+original_hist[1]+original_hist[2])/3)
plt.plot((target_hist[0]+target_hist[1]+target_hist[2])/3)
plt.title('Three Histograms in one figure')
plt.grid('on')
plt.legend(['Specification Histogram','Original Histogram','Target Histogram'])
