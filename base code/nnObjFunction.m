function [obj_val obj_grad] = nnObjFunction(params, n_input, n_hidden, ...
                                    n_class, training_data,...
                                    training_label, lambda)
% nnObjFunction computes the value of objective function (negative log 
%   likelihood error function with regularization) given the parameters 
%   of Neural Networks, thetraining data, their corresponding training 
%   labels and lambda - regularization hyper-parameter.

% Input:
% params: vector of weights of 2 matrices w1 (weights of connections from
%     input layer to hidden layer) and w2 (weights of connections from
%     hidden layer to output layer) where all of the weights are contained
%     in a single vector.
% n_input: number of node in input layer (not include the bias node)
% n_hidden: number of node in hidden layer (not include the bias node)
% n_class: number of node in output layer (number of classes in
%     classification problem
% training_data: matrix of training data. Each row of this matrix
%     represents the feature vector of a particular image
% training_label: the vector of truth label of training images. Each entry
%     in the vector represents the truth label of its corresponding image.
% lambda: regularization hyper-parameter. This value is used for fixing the
%     overfitting problem.
       
% Output: 
% obj_val: a scalar value representing value of error function
% obj_grad: a SINGLE vector of gradient value of error function
% NOTE: how to compute obj_grad
% Use backpropagation algorithm to compute the gradient of error function
% for each weights in weight matrices.
% Suppose the gradient of w1 is 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% reshape 'params' vector into 2 matrices of weight w1 and w2
% w1: matrix of weights of connections from input layer to hidden layers.
%     w1(i, j) represents the weight of connection from unit j in input 
%     layer to unit i in hidden layer.
% w2: matrix of weights of connections from hidden layer to output layers.
%     w2(i, j) represents the weight of connection from unit j in hidden 
%     layer to unit i in output layer.
w1 = reshape(params(1:n_hidden * (n_input + 1)), ...
                 n_hidden, (n_input + 1));

w2 = reshape(params((1 + (n_hidden * (n_input + 1))):end), ...
                 n_class, (n_hidden + 1));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w1_transpose = w1';
w2_transpose = w2';
N = size(training_data,1);

training_data = double(training_data);
training_data = training_data/255;

training_data_with_bias = [training_data (rand(N,1))];



aj = training_data_with_bias*w1_transpose;

zj = sigmoid(aj);

hidden_data_with_bias = [zj (rand(N,1))];
bk = hidden_data_with_bias*w2_transpose;

yk = sigmoid(bk);

%Preparation of target matrix from training label

%tk = zeros(size(yk))+0.01;
tk = zeros(size(yk))+0;

i=1;
while i<N
    %tk(i,training_label(i)+1) = 0.91;
    tk(i,training_label(i)+1) = 1;
    i=i+1;
end

%Computing the Error
error_matrix = ((tk.*log(yk))+((1-tk).*log(1-yk)));

%regularization
reg_cff = (sum(sum(w1_transpose.^2))+sum(sum(w2_transpose.^2)))*lambda/(2*N);
obj_val = ((-1*(sum(sum(error_matrix))))/N)+reg_cff;

%Computing the Gradient of w1 and w2
deltak = yk - tk;

grad_w2 = (((hidden_data_with_bias'*deltak) + (lambda*w2_transpose))/N)';

grad_w1 = training_data_with_bias'*((deltak*w2).*(1- hidden_data_with_bias).*hidden_data_with_bias);
grad_w1 = (( grad_w1(:,1:end-1)+ (lambda*w1_transpose))/N)';

% Suppose the gradient of w1 and w2 are stored in 2 matrices grad_w1 and grad_w2
% Unroll gradients to single column vector
obj_grad = [grad_w1(:) ; grad_w2(:)];

end
