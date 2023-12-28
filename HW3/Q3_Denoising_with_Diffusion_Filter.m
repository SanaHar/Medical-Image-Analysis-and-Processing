%% HW3 
% Sana Harighi- 98104047
%% Q3 
%% Denoising image with defferent filters
clc
clear all;
img = imread("image3.png");
noisy_img = imnoise(img,'gaussian',0,0.01);
figure
subplot(2,3,1)
imshow(img);
title("Original image");

subplot(2,3,2)
imshow(noisy_img);
title("Noisy image");

niter = 25;
kappa = 15;
lambda = 0.15;

aniso_diff1 = anisodiff(noisy_img, niter, kappa, lambda, 1);
aniso_diff2 = anisodiff(noisy_img, niter, kappa, lambda, 2);

constant = 1;
iso_diff = isodiff(noisy_img,lambda,constant);


subplot(2,3,4)
imshow(uint8(aniso_diff1));
title("Denoised image using anisodiff filter - eq1");

subplot(2,3,5)
imshow(uint8(aniso_diff2));
title("Denoised image using anisodiff filter - eq2");

subplot(2,3,6)
imshow(uint8(iso_diff));
title("Denoised image using isodiff filter");
%% calculate SSIM and NIQE
clc
noisy_img = double(noisy_img);
Filter = {'Aniso_diff1';'Aniso_diff2';'iso_diff'};
SSIM = [ssim(aniso_diff1, noisy_img); ssim(aniso_diff2, noisy_img); ssim(iso_diff, noisy_img)];
NIQE = [niqe(aniso_diff1);niqe(aniso_diff2);niqe(iso_diff)];
T = table(Filter,SSIM,NIQE)

%% functions

function ssim = SSIM2(x,y)
k1 = 0.01;
k2 = 0.03;
L = 255;
c1 = (k1*L)^2;
c2 = (k2*L)^2;
mx = mean2(x);
my = mean2(y);
sigmax = std2(x);
sigmay = std2(y);
sigmaxy = corr2(x,y);
num = (2*mx*my+c1)*(2*sigmaxy+c2);
den = (mx^2+my^2+c1)*(sigmax^2+sigmay^2+c2);
ssim = num/den;
end