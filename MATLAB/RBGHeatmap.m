data = load("all_features.mat");

electrode_coord = [4, 8;
                   1, 7;
                   3, 7;
                   2, 6;
                   1, 5;
                   1, 3;
                   4, 1;
                   6, 1;
                   9, 3;
                   9, 5;
                   8, 6;
                   7, 7;
                   9, 7;
                   6, 8];

electrode_coord = electrode_coord + 1;

features = data.all_features;

user11 = features(100,:);
red11 = user11(1:14);
blue11 = user11(15:28);
green11 = user11(29:42);

imred = zeros(11);
imblue = zeros(11);
imgreen = zeros(11);

for i=1:14
    imred(electrode_coord(i,1),electrode_coord(i,2)) = red11(i);
    imblue(electrode_coord(i,1),electrode_coord(i,2)) = blue11(i);
    imgreen(electrode_coord(i,1),electrode_coord(i,2)) = green11(i);
end


imrednew = imresize(imred,[512,512]);
imbluenew = imresize(imblue,[512,512]);
imgreennew = imresize(imgreen,[512,512]);

figure(1)
imagesc(imrednew);
colormap("hot")
colorbar

figure(2)
imagesc(imbluenew);
colorbar
%colormap("hot")

figure(3)
imagesc(imgreennew);
colormap("summer")
colorbar
