% ########################################################################
% #######       DO NOT MODIFY, DEFINITELY READ ALL OF THIS         #######
% ########################################################################

% pkg load image

clear
rand('seed', 100000)
randn('seed', 100000)

%define lenet
layers{1}.type = 'DATA';
layers{1}.height = 28;
layers{1}.width = 28;
layers{1}.channel = 1;
layers{1}.batch_size = 64;

layers{2}.type = 'CONV';
layers{2}.num = 20;
layers{2}.k = 5;
layers{2}.stride = 1;
layers{2}.pad = 0;
layers{2}.group = 1;

layers{3}.type = 'POOLING';
layers{3}.k = 2;
layers{3}.stride = 2;
layers{3}.pad = 0;


layers{4}.type = 'CONV';
layers{4}.k = 5;
layers{4}.stride = 1;
layers{4}.pad = 0;
layers{4}.group = 1;
layers{4}.num = 50;


layers{5}.type = 'POOLING';
layers{5}.k = 2;
layers{5}.stride = 2;
layers{5}.pad = 0;

layers{6}.type = 'IP';
layers{6}.num = 500;
layers{6}.init_type = 'uniform';

layers{7}.type = 'RELU';

layers{8}.type = 'LOSS';
layers{8}.num = 10;


% load data
% change the following value to true to load the entire dataset
fullset = false;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);

xtrain = [xtrain, xvalidate];
ytrain = [ytrain, yvalidate];
m_train = size(xtrain, 2);
batch_size = 64;

% cnn parameters
mu = 0.9;
epsilon = 0.01;
gamma = 0.0001;
power = 0.75;
weight_decay = 0.0005;
w_lr = 1;
b_lr = 2;

test_interval = 500;
display_interval = 10;
snapshot = 250;
max_iter = 10000;

% initialize cnn
params = init_convnet(layers);
param_winc = params;
for l_idx = 1:length(layers)-1
    param_winc{l_idx}.w = zeros(size(param_winc{l_idx}.w));
    param_winc{l_idx}.b = zeros(size(param_winc{l_idx}.b));
end

% learning iterations
for iter = 1 : max_iter
    % get mini-batch and setup the cnn with the mini-batch
    id = randperm(m_train, batch_size);
    [cp, param_grad] = conv_net(params, layers, xtrain(:, id), ytrain(id));

    % we have different epsilons for w and b
    w_rate = get_lr(iter, epsilon*w_lr, gamma, power);
    b_rate = get_lr(iter, epsilon*b_lr, gamma, power);
    [params, param_winc] = sgd_momentum(w_rate, b_rate, mu, weight_decay, params, param_winc, param_grad);

    % display training loss
    if mod(iter, display_interval) == 0
        fprintf('cost = %f training_percent = %f\n', cp.cost, cp.percent);
    end

    % display test accuracy
    if mod(iter, test_interval) == 0
        layers{1}.batch_size = size(xtest, 2);
        [cptest] = conv_net(params, layers, xtest, ytest);
        layers{1}.batch_size = 64;
        fprintf('\ntest accuracy: %f \n\n', cptest.percent);

    end
    
    % save params peridocally to recover from any crashes
    if mod(iter, snapshot) == 0
        filename = 'lenet.mat';
        save(filename, 'params');
    end
end
