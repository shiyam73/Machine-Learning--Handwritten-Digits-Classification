function [train_data, train_label, validation_data, ...
    validation_label, test_data, test_label] = preprocess()
% preprocess function loads the original data set, performs some preprocess
%   tasks, and output the preprocessed train, validation and test data.

% Input:
% Although this function doesn't have any input, you are required to load
% the MNIST data set from file 'mnist_all.mat'.

% Output:
% train_data: matrix of training set. Each row of train_data contains 
%   feature vector of a image
% train_label: vector of label corresponding to each image in the training
%   set
% validation_data: matrix of training set. Each row of validation_data 
%   contains feature vector of a image
% validation_label: vector of label corresponding to each image in the 
%   training set
% test_data: matrix of testing set. Each row of test_data contains 
%   feature vector of a image
% test_label: vector of label corresponding to each image in the testing
%   set

% Some suggestions for preprocessing step:
% - divide the original data set to training, validation and testing set
%       with corresponding labels
% - convert original data set from integer to double by using double()
%       function
% - normalize the data to [0, 1]
% - feature selection

load('mnist_all.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


train = [train0;train1;train2;train3;train4;train5;train6;train7;train8;train9];
test = [test0;test1;test2;test3;test4;test5;test6;test7;test8;test9];

train_and_test = [train;test];
any_nonzeros = find(any(train_and_test,1)==1);

train = train(:,any_nonzeros);
test = test(:,any_nonzeros);

labels0=zeros(size(train0,1),1);
labels1=ones(size(train1,1),1);
labels2=repmat(2,size(train2,1),1);
labels3=repmat(3,size(train3,1),1);
labels4=repmat(4,size(train4,1),1);
labels5=repmat(5,size(train5,1),1);
labels6=repmat(6,size(train6,1),1);
labels7=repmat(7,size(train7,1),1);
labels8=repmat(8,size(train8,1),1);
labels9=repmat(9,size(train9,1),1);

label = [labels0;labels1;labels2;labels3;labels4;labels5;labels6;labels7;labels8;labels9];


r = randperm(60000);

train_data = train(r(1:50000),:);
train_label = label(r(1:50000),:);
validation_data = train(r(50001:end),:);
validation_label = label(r(50001:end),:);
test_data = test;


test_labels0=zeros(size(test0,1),1);
test_labels1=ones(size(test1,1),1);
test_labels2=repmat(2,size(test2,1),1);
test_labels3=repmat(3,size(test3,1),1);
test_labels4=repmat(4,size(test4,1),1);
test_labels5=repmat(5,size(test5,1),1);
test_labels6=repmat(6,size(test6,1),1);
test_labels7=repmat(7,size(test7,1),1);
test_labels8=repmat(8,size(test8,1),1);
test_labels9=repmat(9,size(test9,1),1);

test_label = [test_labels0;test_labels1;test_labels2;test_labels3;test_labels4;test_labels5;test_labels6;test_labels7;test_labels8;test_labels9];

end

