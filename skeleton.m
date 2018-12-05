close all
clear
clc
%% Data Generation Stage
disp('Reading data...')
addpath(genpath('lib'));
% Choose your assigned class
%   -1: Tie Fighter
%   -2: Cloud Car
%   -3: N1 Starfighter
%   -4: Millenium Falcon
my_class = 1;
% Read data
[images,labels] = read_data(my_class);
% Create train and test sets (separate the data)
[Xtrain,Ytrain,Xtest,Ytest] = divide_train_test(images,labels);
disp('Data read complete!')

%% Preprocessing Stage
disp('Preprocessing Stage in progress...')
% Creating empty set of masks
masks = {};

for i = 1:length(Xtrain)
    % Select current image
    I = Xtrain{i};
    %%% Image segmentation: identifying the object
    % Convert I to gray-scale image
    Ig = rgb2gray(I);
    % Obtain the Otsu threshold
    otsu_threshold = graythresh(Ig);
    % Obtain the mask given by this threshold
    mask = imbinarize(Ig, otsu_threshold);
    % Identify object from background (NO NEED TO CODE ANYTHING HERE)
    mask = identify_object(mask);
    
    %%% Morphological processing: improving quality of mask
    % Morphological opening (remove noise)
    se = strel('diamond', 1);
    mask = imopen(mask, se);
    
    % Morphological closing (smooth mask)
    se = strel('diamond', 6);
    mask = imclose(mask, se);
%     imshow(mask,[]);
    
    %%% Storing mask
    masks{end + 1} = mask;

end
disp('Preprocessing complete!')

%% Feature Extraction Stage
disp('Feature Extraction Stage in progress...')
% Creating empy array of features
features = zeros(length(Xtrain),4);

for i = 1:length(Xtrain)
    
    % Select current image
    I = Xtrain{i};
    % Select current mask
    mask = masks{i};

    %%% Feature 1: Colour
    % Convert I to HSV image
    HSV = rgb2hsv(I);
    
%     imshow(HSV,[]);

    % Select Hue component
    H = HSV(:,:,1);
    
    % Select pixels corresponding to preprocessed mask
    relevant_pixels = H(mask);
    
    %imshow(relevant_pixels,[]);

    % Extract colour feature for detected object in current image
    features(i,1) = median(relevant_pixels);
    
    %%% Feature 2: Texture
    % Convert I to gray-scale image
    Ig = rgb2gray(I);
    % Select pixels corresponding to preprocessed mask
    relevant_pixels = Ig(mask);
     
%      imshow(relevant_pixels,[]);
    % Extract texture feature (entropy) for detected object
    features(i,2) = entropy(relevant_pixels);
    
    %%% Feature 3: Shape
    % Obtain object properties (NO NEED TO CODE ANYTHING HERE)
    [area, perimeter, box_area, box_perimeter] = get_object_props(mask);
    % Extract shape feature for detected object
    features(i,3) =  area/box_area;
    features(i,4) =  perimeter/box_perimeter;
    
end
disp('Feature Extraction complete!')

%% Normalization Stage
disp('Normalization Stage in progress...')
% Obtain the mean of each feature
feat_mean = mean(features);
% Obtain the standard deviation of each feature
feat_std  = std(features);
% Normalize the extracted features
features_n = zeros(length(Xtrain),4);

for i=1:size(features,1)
    features_n(i,1) = (features(i,1) - feat_mean(1)) / feat_std(1);
    features_n(i,2) = (features(i,2) - feat_mean(2)) / feat_std(2);
    features_n(i,3) = (features(i,3) - feat_mean(3)) / feat_std(3);
    features_n(i,4) = (features(i,4) - feat_mean(4)) / feat_std(4);
end

% Check if normalization was correctly implemented (VERY IMPORTANT)
% If normalization was correctly implemented, running the line below should
% print the message saying so.
check_normalization(features_n);

% %% Feature Visualization
% % Select pair of features to visualize:
% %   -1: Colour
% %   -2: Texture
% %   -3: Shape
% feat_a = 3 ;
% feat_b = 1 ;
% % Plot feature values in scatter diagram
% figure(1)
% visualize_features(features_n, Ytrain, feat_a, feat_b)

%% Training Stage
disp('Training Stage in progress...')
% Choose the model for the classifier
%   -1: K Nearest Neighbours
%   -2: Support Vector Machine
model_type = 1;

% Generate the model (NO NEED TO CODE ANYTHING HERE)
if model_type == 1
    model = fitcknn(features_n,Ytrain);
else
    model = fitcsvm(features_n,Ytrain,'KernelFunction','rbf','KernelScale','auto');
end
disp('Training completed!');

%% Test Stage
disp('Testing Stage in progress...');
% IMPORTANT!!!
% Test images need to undergo the exact same process as training images

%%% Perform preprocessing (generate masks)
% Creating empty set of masks
masks_test = {};
for i = 1:length(Xtest)
    % Select current image
    I = Xtest{i};
    
  
    
    %%% Image segmentation: identifying the object
    % Convert I to gray-scale image
    Ig = rgb2gray(I);
    % Obtain the Otsu threshold
    otsu_threshold = graythresh(Ig);
    % Obtain the mask given by this threshold
    mask = imbinarize(Ig, otsu_threshold);
    % Identify object from background (NO NEED TO CODE ANYTHING HERE)
    mask = identify_object(mask);
    
    %%% Morphological processing: improving quality of mask
    % Morphological opening (remove noise)
    se = strel('diamond', 1);
    mask = imopen(mask, se);
    
    % Morphological closing (smooth mask)
    se = strel('diamond', 6);
    mask = imclose(mask, se);
    %%% Storing mask
    masks_test{end + 1} = mask;
end
% 
%%% Perform Feature Extraction
% Creating empy array of features
features_test = zeros(length(Xtest),4);
for i = 1:length(Xtest)
    
    % Select current image
    I = Xtest{i};
    % Select current mask
    mask = masks_test{i};
    
    %%% Feature 1: Colour
    % Extract colour feature for detected object in current image
    %features_test(i,1) = .. ;
    
    
    %%% Feature 2: Texture
    % Extract texture feature (entropy) for detected object
    %features_test(i,2) = .. ;
    
    %%% Feature 3: Shape
    % Extract shape feature for detected object
    %features_test(i,3) = ..;


    %%% Feature 1: Colour
    % Convert I to HSV image
    HSV = rgb2hsv(I);
    
%     imshow(HSV,[]);

    % Select Hue component
    H = HSV(:,:,1);
    
    % Select pixels corresponding to preprocessed mask
    relevant_pixels = H(mask);
    
    %imshow(relevant_pixels,[]);

    % Extract colour feature for detected object in current image
    features_test(i,1) = median(relevant_pixels);
    
    %%% Feature 2: Texture
    % Convert I to gray-scale image
    Ig = rgb2gray(I);
    % Select pixels corresponding to preprocessed mask
    relevant_pixels = Ig(mask);
     
%      imshow(relevant_pixels,[]);
    % Extract texture feature (entropy) for detected object
    features_test(i,2) = entropy(relevant_pixels);
    
    %%% Feature 3: Shape
    % Obtain object properties (NO NEED TO CODE ANYTHING HERE)
    [area, perimeter, box_area, box_perimeter] = get_object_props(mask);
    % Extract shape feature for detected object
    features_test(i,3) =  area/box_area;
    features_test(i,4) =  perimeter/box_perimeter;

end

%%% Perform Normalization
% Note that you do not need to recompute the mean and standard deviation
% again. You need to use the values from training.
features_test_n = zeros(length(Xtest),4);

for i=1:size(features_test,1)
    features_test_n(i,1) = (features_test(i,1) - feat_mean(1)) / feat_std(1);
    features_test_n(i,2) = (features_test(i,2) - feat_mean(2)) / feat_std(2);
    features_test_n(i,3) = (features_test(i,3) - feat_mean(3)) / feat_std(3);
    features_test_n(i,4) = (features_test(i,4) - feat_mean(4)) / feat_std(4);
end

%%% Test the model against the new extracted features
% Predict labels (NO NEED TO CODE ANYTHING HERE)
[labels_pred, scores_pred, cost] = predict(model,features_test_n);
scores_pred = scores_pred(:,2);
disp('Testing completed!');

%% Performance Assessment Stage
disp('Performance Assessment Stage in progress...');
labels_true = Ytest';
% Measure the performance of the developed system (P_D and P_FA)
total_pos = 0;
true_pos = 0;
total_neg = 0;
false_pos = 0;
for i=1:size(labels_true,1)
    if labels_true(i) > 0
        total_pos = total_pos + 1;
        if labels_pred(i) == 1
            true_pos = true_pos + 1;
        end
    else
        total_neg = total_neg + 1;
        if labels_pred(i) > 0
            false_pos = false_pos + 1;
        end
    end
end
P_D =  true_pos / total_pos;
P_FA = false_pos / total_neg ;
% Measure the performance of the developed system (AUC)
% (NO NEED TO CODE ANYTHING HERE)
[X,Y,T,AUC] = perfcurve(Ytest',scores_pred,1);
if model_type == 1
    figure(2), plot(X,Y,'LineWidth',3)
else
    figure(2), plot(X,smooth(Y),'LineWidth',3)
end
title(['AUC = ' num2str(AUC)])
disp('Performance Assessed!')
