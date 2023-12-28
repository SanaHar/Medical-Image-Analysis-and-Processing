%% Q4
%% a create dataset matrix and show average image
Xp = [];
img_average = 0;
Cx = 0;
Cy = 0;
d = 120;
for i=1:1:15
  img = imread(strcat(num2str(i),'.jpg'));
  img = imresize(img,[d d]);
  img = im2double(img);
  img_average = img_average+img;
  img = reshape(img,[1,d*d]);
  Xp = [Xp;img];
end
img_average = img_average/15;
figure
imshow(img_average)
title('Average Image')
%% b covariance of Xp
clc
Xp = normalize(Xp);
TF = isnan(Xp);
Xp(TF) = 0;
cov_matrix = cov(Xp);
size(cov_matrix)
[V,D] = eig(cov_matrix);
%% b calculate eigenvalues and eigenvectors 
clc
eigvalues = zeros(d*d,1);
for i=1:1:d*d
    eigvalues(i) = D(i,i);
end
[B,I] = sort(-eigvalues);
eigvalues = eigvalues(I);
eigvectors = V(:, I);
%% c show some eigenvectors
clc
figure
imshow(reshape(eigvectors(1,:),[d,d]))
title('First eigenvector')
figure
imshow(reshape(eigvectors(2,:),[d,d]))
title('Second eigenvector')
%% d reconstruct images using projection
clc
p = 10000;
P = V(:,1:p+1);
projectedMatrix = P * P.';
Xre= Xp*projectedMatrix;
%% show reconstructed images
clc
figure
I = uint8(reshape(Xre(1,:),[d,d]));
imshow(I);
title('recovered Image')
%% b covariance of Xp.T and calculate eigenvalues and eigenvectors 
clc
X = normalize(Xp.');
cov_matrix = cov(X);
size(cov_matrix)
[V,D] = eig(cov_matrix);
eigvalues = zeros(15,1);
for i=1:1:15
    eigvalues(i) = D(i,i);
end

[B,I] = sort(-eigvalues);
eigvalues = eigvalues(I);
eigvectors = V(:, I);

%% c show the first image for number of number eignvectors 1 to 15
clc
Io = reshape(Xp(1,:),[d,d]);
e = zeros(15,1);
figure
for p=1:1:15
P = eigvectors(:,1:p);
projectedMatrix = P * P.';
Xre= projectedMatrix*Xp;
I = reshape(Xre(1,:),[d,d]);
e(p) = sum((Io-I).^2,'all');
subplot(3,5,p);
imshow(I);
title(strcat(num2str(p),' first eignvectors'))
end

figure
plot(e);
title('error of recovered images');
xlabel('num of eigenvectors');
ylabel('error');
%% c show all reconstructed images using 10 eignvectors
clc
e = zeros(15,1);
p = 10;
P = eigvectors(:,1:p);
projectedMatrix = P * P.';
Xre= projectedMatrix*Xp;

figure
for i=1:1:15
    
Io = reshape(Xp(i,:),[d,d]);
I = reshape(Xre(i,:),[d,d]);
e(i) = sum((Io-I).^2,'all');
subplot(3,5,i);
imshow(I);
title(strcat(strcat('reconstructed ',num2str(p)),'.jpg'))
end

figure
plot(e);
title('error of recovered images');
xlabel('num of eigenvectors');
ylabel('error');