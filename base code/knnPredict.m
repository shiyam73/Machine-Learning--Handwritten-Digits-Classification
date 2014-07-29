function label = knnPredict(k, train_data, train_label, test_data)
% knnPredict predicts the label of given data by using k-nearest neighbor
% classification algorithm

% Input:
% k: the parameter k of k-nearest neighbor algorithm
% data: matrix of data. Each row of this matrix represents the feature 
%       vector of a particular image

% Output:
% label: a column vector of predicted labels

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
train_data=double(train_data);
test_data=double(test_data);
[D,I] = pdist2(train_data,test_data,'euclidean','Smallest',k);
predicted_label = train_label(I,:);
predicted_label = reshape(predicted_label,size(I,1),size(I,2));
label = mode(predicted_label);
label = transpose(label);

end

