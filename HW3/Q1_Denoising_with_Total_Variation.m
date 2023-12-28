%% HW3 
% Sana Harighi- 98104047
%% Q1
%% Denoise image
% GPCL
I = imread("TV.png");
f = imnoise(I,'gaussian',0,0.01);
f = double(f);
w1 = zeros(size(f));
w2 = zeros(size(f));
lbd = 0.045;
alpha=0.248;
NIT = 800;
GapTol=0.0001;
verbose = 1;
[ugpcl,w1_gpcl,w2_gpcl,Energy_gpcl,Dgap_gpcl,TimeCost_gpcl,itr_gpcl] = TV_GPCL(w1,w2,f,lbd,alpha,NIT,GapTol,verbose)
w1 = zeros(size(f));
w2 = zeros(size(f));
[uch,w1_ch,w2_ch,Energy_ch,Dgap_ch,TimeCost_ch,itr_ch] = TV_Chambolle(w1,w2,f,lbd,alpha,NIT,GapTol,verbose)

figure
subplot(2,2,1)
imshow(I);
title("Original image");

subplot(2,2,2)
imshow(uint8(f));
title("Noisy image");

subplot(2,2,3)
imshow(uint8(ugpcl));
title("denoised image using TV-GPCL");

subplot(2,2,4)
imshow(uint8(uch));
title("denoised image using TV-Chambolle");

% view w1 and w2
figure
subplot(1,2,1)
imshow(w1_ch);
title("w1 - TV Chambole");

subplot(1,2,2)
imshow(w2_ch);
title("w2 - TV Chambole");

figure
subplot(1,2,1)
imshow(w1_gpcl);
title("w1 - TV GPCL");

subplot(1,2,2)
imshow(w2_gpcl);
title("w2 - TV GPCL");

% calculate snr of denoised image
clc
snr_ch = SNR(f,uch)
snr_gpcl = SNR(f,ugpcl)
%% Observe effect of tolerances on noise reduction for TV-GPCL
clc
w1 = zeros(size(f));
w2 = zeros(size(f));
lbd = 0.045;
alpha=0.248;
NIT = 800;
verbose = 1;
[u_gpcl1,~,~,~,~,~,itr_gpcl1] = TV_GPCL(w1,w2,f,lbd,alpha,NIT,0.01,verbose)
w1 = zeros(size(f));
w2 = zeros(size(f));
[u_gpcl2,~,~,~,~,~,itr_gpcl2]= TV_Chambolle(w1,w2,f,lbd,alpha,NIT,0.0001,verbose)

figure
subplot(2,2,1)
imshow(uint8(f));
title("Noisy image");

subplot(2,2,3)
imshow(uint8(u_gpcl1));
title("denoised image using TV-GPCL, tol=0.01");

subplot(2,2,4)
imshow(uint8(u_gpcl2));
title("denoised image using TV-GPCL, tol=0.0001");


% calculate snr of denoised image
snr_gpcl1 = SNR(f,u_gpcl1)
snr_gpcl2 = SNR(f,u_gpcl2)
%% compare parameters of TV-Chambole and TV-GPCL
clc
figure
plot(Dgap_ch);
hold on
plot(Dgap_gpcl);
legend('Chambole','GPCL')
title("Duality Gap ");
xlabel("num of iterations");
xlim([10, 530])


figure
plot(Energy_ch);
hold on
plot(Energy_gpcl);
legend('Chambole','GPCL')
title("Energy ");
xlabel("num of iterations");


figure
plot(TimeCost_ch);
hold on
plot(TimeCost_gpcl);
legend('Chambole','GPCL')
title("Time Cost");
xlabel("num of iterations");

%% functions
function snr = SNR(f,u)
sum1 = 0;
sum2 = 0;
for i = 1: 1: size(f, 1)
    for j = 1: 1: size(f, 2)
        sum1 = f(i, j)^2 + sum1;
        sum2 = (u(i, j) - f(i,j))^2 + sum2;
    end
end
snr = 10*log10(double(sum1/sum2));
end

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