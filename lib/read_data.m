function [images, labels] = read_data(selected_class)
%read_data Reads the images in the data folder and returns them along with
%the corresponding labels
%   Given an integer input representing each class:
%       -1: Tie Fighter
%       -2: Cloud Car
%       -3: N1 Starfighter
%       -4: Millenium Falcon
%   the function returns a cell array with all the images and a double
%   array with the corresponding labels:
%       +1 for positive class (determined by input)
%       -1 for negative class (determined by input)

images = {};
for i = 1:40
    images{end+1} = imread(['data/1-tie/' num2str(i) '.jpg']);
    images{end+1} = imread(['data/2-cloudcar/' num2str(i) '.jpg']);
    images{end+1} = imread(['data/3-n1/' num2str(i) '.jpg']);
    images{end+1} = imread(['data/4-falcon/' num2str(i) '.jpg']);
end
labels = repmat([1 2 3 4],[1 40]);
labels(labels ~= selected_class) = -1;  
labels(labels == selected_class) =  1;

shuf = randperm(160);
images = images(shuf);
labels = labels(shuf);
end

