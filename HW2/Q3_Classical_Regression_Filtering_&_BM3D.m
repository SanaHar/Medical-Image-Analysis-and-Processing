%% Q3: Classical denoising
%% classical regression filter
clc
clear all;
P = phantom('Modified Shepp-Logan',500);
m = 0;
var_gauss = (0.3);
P_noisy = imnoise(P,'gaussian',m,var_gauss);
imshowpair(P,P_noisy,'montage');

ws = 3;
hx = 0.02*500;
Pf = CR_filter(P_noisy, ws, hx);

figure
imshow(Pf);
title('output after applying classical regression filter');

%% BM3D filter
clc
sigma=0.1;
ws= 3;
search_width= 3; 
l2= 0; 
selection_number = 2;
l3= 1.7;

img = padarray(P_noisy,[search_width search_width],'symmetric','both');
basic_result = first_step(img, sigma, ws, search_width, l2, l3, selection_number);
basic_padded = padarray(basic_result,[search_width search_width],'symmetric','both');
final_result = second_step(img,basic_padded, sigma, ws, search_width, l2, selection_number);

figure
imshow(uint8(final_result));
title('output after applying BM3D filter');

%% functions
function If = CR_filter(I, window_size, hx)

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

function res = dct3(data, mode)
if(nargin > 2 && mode == 'inverse')
    res = idct(data,'dim',3);
    for i = 1:size(data,3)
        res(:,:,i)=idct2(res(:,:,i));
    end
else
    res=zeros(size(data));
    for i=1:size(data,3)
        res(:,:,i)=dct2(data(:,:,i));
    end
    dct(res,'dim',3);
end
end

function basic_result = first_step(I, sigma, ws, sw, l2, l3, sn)
    image_size = size(I);
    num = zeros(size(I));
    den = zeros(size(I));
    bpr = (sw*2+1)-ws+1;
    center_block = bpr^2 / 2 + bpr/2 + 1;
    for i = sw+1:image_size(1)-sw
        for j = sw+1:image_size(2)-sw
            window = I(i-sw:i+sw , j-sw:j+sw);
            blocks = double(im2col(window, [ws ws], 'sliding'));
            dist = zeros(size(blocks,2),1);
            for k = 1:size(blocks,2)
                tmp = wthresh(blocks(:,center_block),'h',sigma*l2)- ...
                    wthresh(blocks(:,k),'h',sigma*l2);
                tmp = reshape(tmp, [ws ws]);
                tmp = norm(tmp,2)^2;
                dist(k) = tmp/(ws^2);
            end
            [~, inds] = sort(dist);
            inds = inds(1:sn);
            blocks = blocks(:, inds);
            p = zeros([ws ws sn]);
            for k = 1:sn
                p(:,:,k) = reshape(blocks(:,k), [ws ws]);
            end
            p = dct3(p);
            p = wthresh(p,'h',sigma*l3);
            wp = 1/sum(p(:)>0);
            p = dct3(p,'inverse');
            for k = 1:sn
                x = max(1,i-sw) + floor((center_block-1)/bpr);
                y = max(1,j-sw) + (mod(center_block-1,bpr));
                num(x:x+ws-1 , y:y+ws-1) = ...
                    num(x:x+ws-1 , y:y+ws-1) + (wp * p(:,:,k));
                den(x:x+ws-1 , y:y+ws-1) = ...
                    den(x:x+ws-1 , y:y+ws-1) + wp;
            end
        end
    end 
    basic_result = num./den;
    basic_result = basic_result(sw+1:end-sw,sw+1:end-sw);
end


function result = second_step(I, basic_res, sigma, ws,sw, l2, sn)
    image_size = size(I);
    num = zeros(size(I));
    den = zeros(size(I));
    bpr = (sw*2+1) - ws + 1; % number of blocks per row/col
    center_block = bpr^2 / 2 + bpr/2 + 1;
    for i = (sw+1):(image_size(1)-sw)
        for j = (sw+1):(image_size(2)-sw)
            window  = I(i-sw:i+sw , j-sw:j+sw);
            window2 = basic_res(i-sw:i+sw, j-sw:j+sw);
            blocks  = double(im2col(window, [ws ws], 'sliding'));
            blocks2 = double(im2col(window2, [ws ws], 'sliding'));
            dist = zeros(size(blocks,2),1);
            for k = 1:size(blocks,2)
                tmp = wthresh(blocks2(:,center_block),'h',sigma*l2)- ...
                    wthresh(blocks2(:,k),'h',sigma*l2);
                tmp = reshape(tmp, [ws ws]);
                tmp = norm(tmp,2)^2;
                dist(k) = tmp/(ws^2);
            end
            [~, I] = sort(dist);
            I = I(1:sn);
            blocks = blocks(:, I);
            blocks2 = blocks2(:, I);
            p = zeros([ws ws sn]);
            basic_p = zeros([ws ws sn]);
            for k = 1:sn
                p(:,:,k) = reshape(blocks(:,k), [ws ws]);
                basic_p(:,:,k) = reshape(blocks2(:,k), [ws ws]);
            end
            basic_p = dct3(basic_p);
            wp = zeros(sn,1);
            for k = 1:sn
                tmp = basic_p(:,:,k);
                tmp = norm(tmp,1).^2;
                wp(k) = tmp/(tmp+(sigma^2));
            end
            p = dct3(p);
            for k = 1:sn
                p(:,:,k) = p(:,:,k)*wp(k);
            end
            p = dct3(p,'inverse');
            wp = 1/sum(wp(:).^2);
            for k = 1:sn
                q = p(:,:,k);
                x = max(1,i-sw) + floor((center_block-1)/bpr);
                y = max(1,j-sw) + (mod(center_block-1,bpr));
                num(x:x+ws-1 , y:y+ws-1) = ...
                    num(x:x+ws-1 , y:y+ws-1) + (wp * q);
                den(x:x+ws-1 , y:y+ws-1) = ...
                    wp + den(x:x+ws-1 , y:y+ws-1);
            end
        end
    end 
    result = num./den;
    result = result(sw+1:end-sw,sw+1:end-sw);
end
