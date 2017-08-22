function p = predict(Theta, X)
% Predict the label of an input given a trained neural network
% with an arbitrary number of hidden layers.
%   p = PREDICT(Theta, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta)
% D.E.Simmons

% Useful values
l = numel(Theta);
m = size(X, 1);
num_labels = size(Theta{l}, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
h{1} = sigmoid([ones(m, 1) X] * transpose(Theta{1}));

for i = 2:l
h{i} = sigmoid([ones(m, 1) h{i-1}] * transpose(Theta{i}));
end
[dummy, p] = max(h{l}, [], 2);


function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

g = 1.0 ./ (1.0 + exp(-z));
