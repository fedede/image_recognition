function [Xtrain,Ytrain,Xtest,Ytest] = divide_train_test(images,labels)
%divide_train_test Divides the input data in two sets for training and
%testing
%   The function makes sure that there is always at least 6 positive
%   samples for testing
enough_test = false;
while ~enough_test
    train_ptg = 0.7;
    N = length(images); 
    tf = false(1,N);   
    tf(1:round(train_ptg*N)) = true;  
    tf = tf(randperm(N));   
    Xtrain = images(tf); 
    Xtest = images(~tf);
    Ytrain = labels(tf); 
    Ytest = labels(~tf);
    if sum(Ytest==1) > 5
        enough_test = true;
    end
end
end

