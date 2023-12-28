%% Q4
%% add noise to image
clc
clear all;
img = imread("image_anisotropic.png");
noisy_img = imnoise(img,'gaussian',0,0.01);

figure
subplot(1, 2, 1)
imshow(img)
title('input image')
subplot(1, 2, 2)
imshow(noisy_img)
title('noisy image')
%% effect of lambda and equation
clc
noisy_img = double(noisy_img);
niter = 10;
kappa =10;
option = 1;
figure
i = 1;
snr = zeros(1,6);
for lbd = 0:0.05:0.25
    subplot(2,3,i);
    denoised_img = anisodiff(noisy_img, niter, kappa, lbd, option);
    imshow(uint8(denoised_img))
    title(strcat("denoised image using anisodiff filter and eq1, ",num2str(lbd)))
    snr(i) = SNR(noisy_img,denoised_img);
    i = i+1;
end
 
figure
plot(0:0.05:0.25,snr);
title("snr of denoised image using anisodiff filter and eq1");
xlabel("lbd");
ylabel("snr");

option = 2;
figure
i = 1;
snr = zeros(1,6);
for lbd = 0:0.05:0.25
    subplot(2,3,i);
    denoised_img = anisodiff(noisy_img, niter, kappa, lbd, option);
    imshow(uint8(denoised_img))
    title(strcat("denoised image using anisodiff filter and eq2, ",num2str(lbd)))
    snr(i) = SNR(noisy_img,denoised_img);
    i = i+1;
end
 
figure
plot(0:0.05:0.25,snr);
title("snr of denoised image using anisodiff filter and eq2");
xlabel("lbd");
ylabel("snr");

%% effect of kappa and equation
clc
noisy_img = double(noisy_img);
niter = 10;
lbd =0.1;
option = 1;
figure
i = 1;
snr = zeros(1,6);
for kappa = 0:5:25
    subplot(2,3,i);
    denoised_img = anisodiff(noisy_img, niter, kappa, lbd, option);
    imshow(uint8(denoised_img))
    title(strcat("denoised image using anisodiff filter and eq1, ",num2str(kappa)))
    snr(i) = SNR(noisy_img,denoised_img);
    i = i+1;
end
 
figure
plot(0:5:25,snr);
title("snr of denoised image using anisodiff filter and eq1");
xlabel("kappa");
ylabel("snr");

option = 2;
figure
i = 1;
snr = zeros(1,6);
for kappa = 0:5:25
    subplot(2,3,i);
    denoised_img = anisodiff(noisy_img, niter, kappa, lbd, option);
    imshow(uint8(denoised_img))
    title(strcat("denoised image using anisodiff filter and eq2, ",num2str(kappa)))
    snr(i) = SNR(noisy_img,denoised_img);
    i = i+1;
end
 
figure
plot(0:5:25,snr);
title("snr of denoised image using anisodiff filter and eq2");
xlabel("kappa");
ylabel("snr");

%% effect of niter and choose best denoised image
clc
noisy_img = double(noisy_img);
niter = 10;
lbd =0.1;
option = 1;
kappa = 10;
figure
denoised_img = anisodiff(noisy_img, niter, kappa, lbd, option);
imshow(uint8(denoised_img))
title("denoised image using anisodiff filter and eq1, niter=10000")
snr = SNR(noisy_img,denoised_img)
imwrite(denoised_img,'anisotropic_best.png');
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


function diff = anisodiff(im, niter, kappa, lambda, option)

if ndims(im)==3
  error('Anisodiff only operates on 2D grey-scale images');
end

im = double(im);
[rows,cols] = size(im);
diff = im;
  
for i = 1:niter
  diffl = zeros(rows+2, cols+2);
  diffl(2:rows+1, 2:cols+1) = diff;

  % North, South, East and West differences
  deltaN = diffl(1:rows,2:cols+1)   - diff;
  deltaS = diffl(3:rows+2,2:cols+1) - diff;
  deltaE = diffl(2:rows+1,3:cols+2) - diff;
  deltaW = diffl(2:rows+1,1:cols)   - diff;

  % Conduction
  if option == 1
    cN = exp(-(deltaN/kappa).^2);
    cS = exp(-(deltaS/kappa).^2);
    cE = exp(-(deltaE/kappa).^2);
    cW = exp(-(deltaW/kappa).^2);
  elseif option == 2
    cN = 1./(1 + (deltaN/kappa).^2);
    cS = 1./(1 + (deltaS/kappa).^2);
    cE = 1./(1 + (deltaE/kappa).^2);
    cW = 1./(1 + (deltaW/kappa).^2);
  end

  diff = diff + lambda*(cN.*deltaN + cS.*deltaS + cE.*deltaE + cW.*deltaW);

end
end

