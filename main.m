clear all
% This script loads training and testing data, and
% calls the function neurnet, which calculates the 
% trained neural network parameters. The script
% then tests the parameters using predict and display data.
% The functions are a generaliztion of those that were
% built as part of the Coursera machine learning course 
% by Andrew Ng. The generalization allows the user to 
% specify an arbitray number of hidden layers using "layer_sizes".
% 
% D.E.Simmons
load('data.mat');  % load training data X and y

layer_sizes = [size(X,2) 40 20];% Input layer and hidden layer sizes. 
                            % Input layer must depend on loaded data.
                            % General would be 
                            % layer_sizes = [size(X,2) hiddenL1 ... hiddenLn]
num_labels = max(y);        % Number of output labels, which must depend
                            % on loaded data. 

lambda = 1;           % regularization parameter
frac = 2/3;           % fraction of data used to train NN
m = size(X, 1);
rp = randperm(m);

Theta = neurnet(layer_sizes, num_labels, X(rp(1:floor(size(X, 1)*frac)),:) ,...
         y(rp(1:floor(size(X, 1)*frac))), lambda); %train the neural network

error = 0;

for i = rp(floor(size(X, 1)*frac)+1:end)
    pred = predict(Theta, X(i,:));
    fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, y(i))
    disp('(Press Enter)')
    displayData(X(i,:));
    pause
    error = (pred == y(i)) + error;   
end
error/length(rp(floor(size(X, 1)*frac)+1:end))