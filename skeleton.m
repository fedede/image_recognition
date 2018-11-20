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
    Ig = rgb2gray(I) ;
    % Obtain the Otsu threshold
    otsu_threshold = graythresh(Ig) ;
    % Obtain the mask given by this threshold
%     imshow(Ig, []); 
    mask = imbinarize(Ig, otsu_threshold);
    % Identify object from background (NO NEED TO CODE ANYTHING HERE)
    mask = identify_object(mask);
    imshow(mask, []);
    %%% Morphological processing: improving quality of mask
    % Morphological opening (remove noise)
    se =  strel('diamond', 1);
    % Morphological closing (smooth mask)
    mask =  imopen(mask, se);
    imshow(mask,[]);
    %%% Storing mask
    se = strel('diamond', 6);
    mask = imclose(mask, se);
    imshow(mask,[]);
    masks{end + 1} = mask;

end
disp('Preprocessing complete!')

%% Feature Extraction Stage
% disp('Feature Extraction Stage in progress...')
% % Creating empy array of features
% features = zeros(length(Xtrain),3);
% 
% for i = 1:length(Xtrain)
%     
%     % Select current image
%     I = Xtrain{i};
%     % Select current mask
%     mask = masks{i};
%     
%     %%% Feature 1: Colour
%     % Convert I to HSV image
%     HSV = .. ;
%     % Select Hue component
%     H = HSV(:,:,1);
%     % Select pixels corresponding to preprocessed mask
%     relevant_pixels = H(mask);
%     % Extract colour feature for detected object in current image
%     features(i,1) = .. ;
%     
%     %%% Feature 2: Texture
%     % Convert I to gray-scale image
%     Ig = .. ;
%     % Select pixels corresponding to preprocessed mask
%     relevant_pixels = .. ;
%     % Extract texture feature (entropy) for detected object
%     features(i,2) = .. ;
%     
%     %%% Feature 3: Shape
%     % Obtain object properties (NO NEED TO CODE ANYTHING HERE)
%     [area, perimeter] = get_object_props(mask);
%     % Extract shape feature for detected object
%     features(i,3) = .. ;
%     
% end
% disp('Feature Extraction complete!')
% 
% %% Normalization Stage
% disp('Normalization Stage in progress...')
% % Obtain the mean of each feature
% feat_mean = .. ;
% % Obtain the standard deviation of each feature
% feat_std  = .. ;
% % Normalize the extracted features
% features_n = .. ;
% 
% % Check if normalization was correctly implemented (VERY IMPORTANT)
% % If normalization was correctly implemented, running the line below should
% % print the message saying so.
% check_normalization(features_n);
% 
% %% Feature Visualization
% % Select pair of features to visualize:
% %   -1: Colour
% %   -2: Texture
% %   -3: Shape
% feat_a = .. ;
% feat_b = .. ;
% % Plot feature values in scatter diagram
% figure(1)
% visualize_features(features_n, Ytrain, feat_a, feat_b)
% 
% %% Training Stage
% disp('Training Stage in progress...')
% % Choose the model for the classifier
% %   -1: K Nearest Neighbours
% %   -2: Support Vector Machine
% model_type = ;
% 
% % Generate the model (NO NEED TO CODE ANYTHING HERE)
% if model_type == 1
%     model = fitcknn(features_n,Ytrain);
% else
%     model = fitcsvm(features_n,Ytrain,'KernelFunction','rbf','KernelScale','auto');
% end
% disp('Training completed!')
% 
% %% Test Stage
% disp('Testing Stage in progress...')
% % IMPORTANT!!!
% % Test images need to undergo the exact same process as training images
% 
% %%% Perform preprocessing (generate masks)
% % Creating empty set of masks
% masks_test = {};
% for i = 1:length(Xtest)
%     % Select current image
%     I = Xtest{i};
%     
%     %%% Image segmentation: identifying the objecfeatures_testt
%     % CODE GOES HERE
%     %%% Morphological processing: improving quality of mask
%     % CODE GOES HERE
%     
%     %%% Storing mask
%     masks_test{end + 1} = mask;
% end
% 
% %%% Perform Feature Extraction
% % Creating empy array of features
% features_test = zeros(length(Xtest),3);
% for i = 1:length(Xtest)
%     
%     % Select current image
%     I = Xtest{i};
%     % Select current mask
%     mask = masks_test{i};
%     
%     %%% Feature 1: Colour
%     % Extract colour feature for detected object in current image
%     features_test(i,1) = .. ;
%     
%     %%% Feature 2: Texture
%     % Extract texture feature (entropy) for detected object
%     features_test(i,2) = .. ;
%     
%     %%% Feature 3: Shape
%     % Extract shape feature for detected object
%     features_test(i,3) = ..;
%     
% end
% 
% %%% Perform Normalization
% % Note that you do not need to recompute the mean and standard deviation
% % again. You need to use the values from training.
% features_test_n = .. ;
% 
% %%% Test the model against the new extracted features
% % Predict labels (NO NEED TO CODE ANYTHING HERE)
% [labels_pred, scores_pred, cost] = predict(model,features_test_n);
% scores_pred = scores_pred(:,2);
% disp('Testing completed!')
% 
% %% Performance Assessment Stage
% disp('Performance Assessment Stage in progress...')
% labels_true = Ytest';
% % Measure the performance of the developed system (P_D and P_FA)
% P_D = .. ;
% P_FA = .. ;
% % Measure the performance of the developed system (AUC)
% % (NO NEED TO CODE ANYTHING HERE)
% [X,Y,T,AUC] = perfcurve(Ytest',scores_pred,1);
% if model_type == 1
%     figure(2), plot(X,Y,'LineWidth',3)
% else
%     figure(2), plot(X,smooth(Y),'LineWidth',3)
% end
% title(['AUC = ' num2str(AUC)])
% disp('Performance Assessed!')
