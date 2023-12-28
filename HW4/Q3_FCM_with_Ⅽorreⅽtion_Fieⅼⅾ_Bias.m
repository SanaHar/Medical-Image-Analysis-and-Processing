%% Q3.2
%% 2.1
clc
load('imageData.mat');
load('imageMask.mat');
imwrite(data,'img.jpg','JPG');
imwrite(mask,'mask.jpg','JPG');
img = data;
%% 2.2
clc
k = 3;
q = 5;
bconst = 1;
binit = (bconst.*ones(size(mask))).*mask;
f = 9;
sigma = 2.5;
[x,y]=meshgrid(-4:4,-4:4);
Exp_comp = -(x.^2+y.^2)/(2*sigma*sigma);
kernel= exp(Exp_comp)/(2*pi*sigma*sigma);
%% 2.3
clc
eps = 0.0000001;
[seg, mu, sigma] = KMeans(img, mask, k, eps);
showSegmented(seg, k, 'segmented image after Kmeans' , 'C:\Users\Beethoven\Desktop\hw4');
%% 2.4.1
clc
eps = 0.00001;
N_max = 200;
J_init = 100000000;
[u, b, c, J] = iterate(img, mask, seg, binit, mu, q, kernel, J_init, eps, N_max);
%% 2.4.2
plot(1:1:200,J')
title('Objective Function for different number of iterations')
xlabel('Iteration')
%% 2.4.3
figure
imshow(b)
title('Bias Field')
%% 2.4.4
clc
map = [0 0 0; 1 0 0; 0 1 0; 0 0 1];

figure
subplot(1,3,1);
imshow(img);
title('Corrupted Image');

subplot(1,3,2);
img_br = computeA(u, c);
imshow(img_br);
title('Bias Removed Image');

subplot(1,3,3);
img_r = img-binit;
imshow(img_r);
title('Residual Image');

figure
subplot(1,2,1);
imshow(seg/k);
title('Segmented Image before applying algorithm');

subplot(1,2,2);
imshow(u);
title('Segmented Image after applying algorithm');
colormap(map);
    