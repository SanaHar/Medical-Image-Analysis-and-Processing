%% Q1 : non local mean filter
%% read images and add gaussian and salt & pepper noise to them separately
clc
clear all;
I1 = imread('image1.png');
I2 = imread('image2.png');
I1 = rgb2gray(I1);
I2 = rgb2gray(I2);
m = 0.5;
var_gauss = 0.1;
d = 0.01;
I1_g = imnoise(I1,'gaussian',m,var_gauss);
I2_g = imnoise(I2,'gaussian',m,var_gauss);
I1_sp = imnoise(I1,'salt & pepper',d);
I2_sp = imnoise(I2,'salt & pepper',d);

figure 
subplot(1,3,1);
imshow(I1);
title('First original image');
subplot(1,3,2);
imshow(I1_g);
title('First image with gaussian noise');
subplot(1,3,3);
imshow(I1_sp);
title('First image with slat & pepper noise');

figure 
subplot(1,3,1);
imshow(I2);
title('Second original image');
subplot(1,3,2);
imshow(I2_g);
title('Second image with gaussian noise');
subplot(1,3,3);
imshow(I2_sp);
title('Second image with slat & pepper noise');
%% compare NLM and NLM2
clc
sigma = 0.1;
wsim = 3;
w = 5;
If1 = NLM(I1_g,sigma,wsim,w); 
p1 = PSNR(I1,If1,255)
If2 = NLM2(I1_g,sigma,wsim,w); 
p2 = PSNR(I1,If2,255)
figure
subplot(1,2,1);
imshow(uint8(If1));
title('result of NLM');
subplot(1,2,2);
imshow(uint8(If2));
title('result of NLM2');
%% apply NLM filter with Wsim=W=3
clc
wsim = 3;
w = 3;
p1_g = zeros(1,5);
p1_sp = zeros(1,5);
p2_g = zeros(1,5);
p2_sp = zeros(1,5);
i = 1;
for sigma = 0.1:0.1:0.5
    If = NLM2(I1_g,sigma,wsim,w); 
    p1_g(i) = PSNR(I1,If,255);
    If = NLM2(I1_sp,sigma,wsim,w); 
    p1_sp(i) = PSNR(I1,If,255);
    If = NLM2(I2_g,sigma,wsim,w); 
    p2_g(i) = PSNR(I2,If,255);
    If = NLM2(I2_sp,sigma,wsim,w); 
    p2_sp(i) = PSNR(I2,If,255);
    i = i+1;
end
%% show result of first NLM filter
clc
figure
plot(0.1:0.1:0.5,p1_g);
title('psnr of image1 with gaussian noise after applying NLM filter with w=3 and wsim=3');
ylabel('psnr in db');
xlabel('sigma');

figure
plot(0.1:0.1:0.5,p1_sp);
title('psnr of image1 with salt & pepper noise after applying NLM filter with w=3 and wsim=3');
ylabel('psnr in db');
xlabel('sigma');

figure
plot(0.1:0.1:0.5,p2_g);
title('psnr of image2 with gaussian noise after applying NLM filter with w=3 and wsim=3');
ylabel('psnr in db');
xlabel('sigma');

figure
plot(0.1:0.1:0.5,p2_sp);
title('psnr of image2 with salt & pepper noise after applying NLM filter with w=3 and wsim=3');
ylabel('psnr in db');
xlabel('sigma');


%% apply NLM filter with Wsim=3 and W=5
clc
wsim = 3;
w = 5;
pn1_g = zeros(1,5);
pn1_sp = zeros(1,5);
pn2_g = zeros(1,5);
pn2_sp = zeros(1,5);
i = 1;
for sigma = 0.1:0.1:0.5
    If = NLM2(I1_g,sigma,wsim,w); 
    pn1_g(i) = PSNR(I1,If,255);
    If = NLM2(I1_sp,sigma,wsim,w); 
    pn1_sp(i) = PSNR(I1,If,255);
    If = NLM2(I2_g,sigma,wsim,w); 
    pn2_g(i) = PSNR(I2,If,255);
    If = NLM2(I2_sp,sigma,wsim,w); 
    pn2_sp(i) = PSNR(I2,If,255);
    i = i+1;
end
%% show result of second NLM filter
clc
figure
plot(0.1:0.1:0.5,pn1_g);
title('psnr of image1 with gaussian noise after applying NLM filter with w=5 and wsim=3');
ylabel('psnr in db');
xlabel('sigma');

figure
plot(0.1:0.1:0.5,pn1_sp);
title('psnr of image1 with salt & pepper noise after applying NLM filter with w=5 and wsim=3');
ylabel('psnr in db');
xlabel('sigma');

figure
plot(0.1:0.1:0.5,pn2_g);
title('psnr of image2 with gaussian noise after applying NLM filter with w=5 and wsim=3');
ylabel('psnr in db');
xlabel('sigma');

figure
plot(0.1:0.1:0.5,pn2_sp);
title('psnr of image2 with salt & pepper noise after applying NLM filter with w=5 and wsim=3');
ylabel('psnr in db');
xlabel('sigma');

%% apply gaussian filter 
clc
pg1_g = zeros(1,5);
pg1_sp = zeros(1,5);
pg2_g = zeros(1,5);
pg2_sp = zeros(1,5);
kernel_size = 77;
i = 1;
for sigma = 0.1:0.1:0.5
    If = gaussian_filter(I1_g, sigma, kernel_size);
    pg1_g(i) = PSNR(I1,If,255);
    If = gaussian_filter(I1_sp, sigma, kernel_size); 
    pg1_sp(i) = PSNR(I1,If,255);
    If = gaussian_filter(I2_g, sigma, kernel_size);
    pg2_g(i) = PSNR(I2,If,255);
    If = gaussian_filter(I2_sp, sigma, kernel_size);
    pg2_sp(i) = PSNR(I2,If,255);
    i = i+1;
end
%% show result of gaussian filter
clc
figure
plot(0.1:0.1:0.5,pg1_g);
hold on
plot(0.1:0.1:0.5,pn1_g);
hold on
plot(0.1:0.1:0.5,p1_g);
title('psnr of image1 with gaussian noise after applying gaussian filter');
ylabel('psnr in db');
xlabel('sigma');
legend('gaussian filter','NLM with w=5, wsim=3', 'NLM with w=3, wsim=3');

figure
plot(0.1:0.1:0.5,pg1_sp);
hold on
plot(0.1:0.1:0.5,pn1_sp);
hold on
plot(0.1:0.1:0.5,p1_sp);
title('psnr of image1 with salt & pepper noise after applying gaussian filter');
ylabel('psnr in db');
xlabel('sigma');
legend('gaussian filter','NLM with w=5, wsim=3', 'NLM with w=3, wsim=3');

figure
plot(0.1:0.1:0.5,pg2_g);
hold on
plot(0.1:0.1:0.5,pn2_g);
hold on
plot(0.1:0.1:0.5,p2_g);
title('psnr of image2 with gaussian noise after applying gaussian filter');
ylabel('psnr in db');
xlabel('sigma');
legend('gaussian filter','NLM with w=5, wsim=3', 'NLM with w=3, wsim=3');

figure
plot(0.1:0.1:0.5,pg2_sp);
hold on
plot(0.1:0.1:0.5,pn2_sp);
hold on
plot(0.1:0.1:0.5,p2_sp);
title('psnr of image2 with salt & pepper noise after applying gaussian filter');
ylabel('psnr in db');
xlabel('sigma');
legend('gaussian filter','NLM with w=5, wsim=3', 'NLM with w=3, wsim=3');
%% show best denoised images based on psnr for both filters
clc
If1_g_nlm = NLM2(I1_g,0.5,3,5); 
PSNR(I1,If1_g_nlm,255)
If2_g_nlm = NLM2(I2_g,0.5,3,5); 
PSNR(I2,If2_g_nlm,255)

If1_g_g = gaussian_filter(I1_g, 0.5, 77);
PSNR(I1,If1_g_g,255)
If2_g_g = gaussian_filter(I2_g, 0.5, 77);
PSNR(I2,If2_g_g,255)

If1_sp_g = gaussian_filter(I1_sp, 0.5, 77);
PSNR(I1,If1_sp_g,255)
If2_sp_g = gaussian_filter(I2_sp, 0.5, 77);
PSNR(I2,If2_sp_g,255)

If1_sp_nlm = NLM2(I1_sp,0.1,3,3);
PSNR(I1,If1_sp_nlm,255)
If2_sp_nlm = NLM2(I2_sp,0.5,3,3);
PSNR(I2,If2_sp_nlm ,255)

figure
subplot(1,3,1);
imshow(I1_g);
title('image1 with gaussian noise');
subplot(1,3,2);
imshow(uint8(If1_g_g));
title('filtered image with gaussian filter');
subplot(1,3,3);
imshow(uint8(If1_g_nlm));
title('filtered image with nlm filter');

figure
subplot(1,3,1);
imshow(I2_g);
title('image2 with gaussian noise');
subplot(1,3,2);
imshow(uint8(If2_g_g));
title('filtered image with gaussian filter');
subplot(1,3,3);
imshow(uint8(If2_g_nlm));
title('filtered image with nlm filter');

figure
subplot(1,3,1);
imshow(I1_sp);
title('image1 with salt & pepper noise');
subplot(1,3,2);
imshow(uint8(If1_sp_g));
title('filtered image with gaussian filter');
subplot(1,3,3);
imshow(uint8(If1_sp_nlm));
title('filtered image with nlm filter');


figure
subplot(1,3,1);
imshow(I2_sp);
title('image2 with salt & pepper noise');
subplot(1,3,2);
imshow(uint8(If2_sp_g));
title('filtered image with gaussian filter');
subplot(1,3,3);
imshow(uint8(If2_sp_nlm));
title('filtered image with nlm filter');
%% functions
function p = PSNR(I,If,L)
[M,N] = size(I);
I = double(I);
p = 10*log10(L*L*M*N/sum((I-If).^2,'all'));
end

% Function to calculate new pixel intensity after application
% of non local means filter at pixel
function finalvalue = NLM_filt(pi,pj,sigma,I,Gp,wsim,w)

SgX = I(pi-wsim:pi+wsim, pj-wsim:pj+wsim);
C = 0;
weighted_sum=0;

for i = pi-w:pi+w
    for j = pj-w:pj+w
        SgY = I(i-wsim:i+wsim,j-wsim:j+wsim); 
        dist = sum(Gp.*((SgX-SgY).^2),'all');
        Kxy = exp(-dist/sigma^2);
        C = C + Kxy;
        weighted_sum = weighted_sum + Kxy*I(i,j);    
    end
end
finalvalue = weighted_sum/C;  
%finalvalue = round(finalvalue); 
end

function If = NLM(I,sigma,w,wsim)
[N,M] = size(I);
I = double(I);

If = zeros(N,M); 
pad_req = w+wsim;
I_pad = padarray(I, [pad_req pad_req], 'replicate');

mask = zeros(2*wsim+1,2*wsim+1);

for i = 1:2*wsim+1
    for j = 1:2*wsim+1
        mask(i,j) = exp(-((i-wsim)^2+(j-wsim)^2)/((2*wsim+1)*(2*wsim+1)));
    end 
end
mask = mask./sum(mask,'all');

for i = 1:N
    for j = 1:M
        If(i,j) = NLM_filt(i+pad_req,j+pad_req,sigma,I_pad,mask,wsim,w);
    end
end
end


function If = gaussian_filter(I, sigma, kernel_size)
kernel = zeros(kernel_size,kernel_size);
w = 0;
miu = (kernel_size+1)/2;
 for i=1:1:kernel_size
     for j=1:1:kernel_size
         sq_dist = (i-miu)^2+(j-miu)^2;
         kernel(i,j) = exp(-1*(sq_dist)/(2*sigma*sigma));
         w = w+kernel(i,j);
     end
 end
 kernel = kernel/w;
 [m,n] = size(I);
 If = zeros(m,n);
 Im = padarray(I,[(kernel_size-1)/2 (kernel_size-1)/2]);
 
 for i=1:1:m
     for j=1:1:n
         temp = Im(i:i+kernel_size-1, j:j+kernel_size-1);
         temp = double(temp);
         conv = temp.*kernel;
         If(i,j) = sum(conv(:));
     end
 end
end

function finalvalue = NLM_filt2(pi,pj,sigma,I,wsim,w)

Bp = I(pi-wsim:pi+wsim, pj-wsim:pj+wsim);
Bp = mean(Bp,'all');
C = 0;
weighted_sum=0;

for i = pi-w:pi+w
    for j = pj-w:pj+w
        Bq = I(i-wsim:i+wsim,j-wsim:j+wsim);
        Bq = mean(Bq,'all');
        %dist = sum((Bp-Bq).^2,'all');
        dist = (Bp-Bq)^2;
        F = exp(-dist/(sigma^2));
        C = C + F;
        weighted_sum = weighted_sum + F*I(i,j);    
    end
end
finalvalue = weighted_sum/C;
end

function If = NLM2(I,sigma,w,wsim)
[N,M] = size(I);
I = double(I);

If = zeros(N,M); 
pad_req = w+wsim;
I_pad = padarray(I, [pad_req pad_req], 'replicate');

for i = 1:N
    for j = 1:M
        If(i,j) = NLM_filt2(i+pad_req,j+pad_req,sigma,I_pad,wsim,w);
    end
end
end
