%% Q2 : bilateral filter
%%
clc 
clear all;
ws = 3;
I = imread('q2.png');
[n,m,~] = size(I);
I = im2double(I);

hx  = 0.02*max(n,m);
hg = 0.05*(max(I,[],'all') - min(I,[],'all'));

Id = zeros(size(I));
for i=1:1:3
Id(:,:,i) = filter_distance(I(:,:,i), ws, hx);
end
figure
subplot(1,2,1);
imshow(Id);
title('output after applying distance kernel');

Ig =  zeros(size(I));
for i=1:1:3
Ig(:,:,i) = filter_gray_level(I(:,:,i), ws, hg);
end
subplot(1,2,2);
imshow(Ig);
title('output after applying gray level kernel');
S = size(I);
If =  zeros(size(I));
Ikernel = zeros(5,ws,ws,3);
Isub =  zeros(5,ws,ws,3);
for i=1:1:3
[If(:,:,i), Ikernel(:,:,:,i), Isub(:,:,:,i)] =  bilateral_filter(I(:,:,i), ws, hg, hx);
end
figure
imshow(If);
title('output after applying both distance and gray level kernels');

figure
for i=1:1:5
subplot(2,3,i);
II = [Ikernel(i,:,:,1); Ikernel(i,:,:,2); Ikernel(i,:,:,3)];
imshow(II);
title(strcat('Kerenl of bilateral filter for part ',num2str(i)));
end 

figure
for i=1:1:5
subplot(2,3,i);
II = [Isub(i,:,:,1); Isub(i,:,:,2); Isub(i,:,:,3)];
imshow(II);
title(strcat('Sub image of bilateral filter for part ',num2str(i)));
end 
%% functions 
function [If,Ikernel,Isub] = bilateral_filter(I, window_size, hg, hx)

[n, m] = size(I);
pad_req = (window_size -1)/2;
I_pad = padarray(I, [pad_req, pad_req], 'both', 'replicate');
If = zeros([n, m]);
[x, y] = meshgrid(-pad_req:pad_req, -pad_req:pad_req);
Ghx = exp(-(x-y).^2/(2*hx^2));
k = 1;
S = size(Ghx);
Ikernel = zeros(5,S(1),S(2));
Isub = zeros(5,S(1),S(2));

for i = 1+pad_req:n+pad_req
    
    for j = 1+pad_req:m+pad_req
        
        sub_img = I_pad(i-pad_req : i+pad_req, j-pad_req : j+pad_req);
        Ghy = exp(-(sub_img-I_pad(i,j)).^2/(2*hg^2));
        kernel = Ghy.*Ghx;
        norm = sum(kernel(:));
        If(i-pad_req, j-pad_req) = sum(sum(kernel.*sub_img))/norm;
    end
    if k<6
        Ikernel(k,:,:) = kernel;
        Isub(k,:,:) = sub_img;
        k = k+1;
    end
end

end

function If = filter_gray_level(I, window_size, hg)

[n, m] = size(I);
pad_req = (window_size-1)/2;
I_pad = padarray(I, [pad_req, pad_req], 'both', 'replicate');
If = zeros([n, m]);

for i = 1+pad_req:n+pad_req
    for j = 1+pad_req:m+pad_req
        
        sub_img = I_pad(i-pad_req : i+pad_req, j-pad_req : j+pad_req);
        Ghy = exp(-(sub_img-I_pad(i,j)).^2/(2*hg^2));
        kernel = Ghy;
        norm = sum(kernel(:));
        If(i-pad_req, j-pad_req) = sum(sum(kernel.*sub_img))/norm;
        
    end
end

end


function If = filter_distance(I, window_size, hx)

[n, m] = size(I);
pad_req = (window_size-1)/2;
I_pad = padarray(I, [pad_req, pad_req], 'both', 'replicate');
If = zeros([n, m]);
[x, y] = meshgrid(-pad_req:pad_req, -pad_req:pad_req);
Ghx = exp(-(x-y).^2/(2*hx^2));

for i = 1+pad_req:n+pad_req
    for j = 1+pad_req:m+pad_req
        
        sub_img = I_pad(i-pad_req : i+pad_req, j-pad_req : j+pad_req);
        kernel = Ghx;
        norm = sum(kernel(:));
        If(i-pad_req, j-pad_req) = sum(sum(kernel.*sub_img))/norm;
        
    end
end

end
