% ########################################################################
% #######     DO NOT MODIFY, READ ONLY IF YOU ARE INTERESTED       #######
% ########################################################################

function [cp, param_grad] = conv_net(params, layers, data, labels)
% Args:
% params: a cell array that stores hyper parameters
% layers: a cell array that defines LeNet
% data: input data with shape (784, batch size)
% labels: label with shape (batch size)

% Returns:
% cp: train accuracy for the train data
% param_grad: gradients of all the layers whose parameters are stored in params

l = length(layers);
batch_size = layers{1}.batch_size;
assert(strcmp(layers{1}.type, 'DATA') == 1, 'first layer must be data layer');
output{1}.data = data;
output{1}.height = layers{1}.height;
output{1}.width = layers{1}.width;
output{1}.channel = layers{1}.channel;
output{1}.batch_size = layers{1}.batch_size;
output{1}.diff = 0;
for i = 2:l-1
    switch layers{i}.type
        case 'CONV'
            output{i} = conv_layer_forward(output{i-1}, layers{i}, params{i-1});
        case 'POOLING'
            output{i} = pooling_layer_forward(output{i-1}, layers{i});
        case 'IP'
            output{i} = inner_product_forward(output{i-1}, layers{i}, params{i-1});
        case 'RELU'
            output{i} = relu_forward(output{i-1}, layers{i});
        case 'ELU'
            output{i} = elu_forward(output{i-1}, layers{i});
    end
end
i = l;
assert(strcmp(layers{i}.type, 'LOSS') == 1, 'last layer must be loss layer');

wb = [params{i-1}.w(:); params{i-1}.b(:)];
[cost, grad, input_od, percent] = mlrloss(wb, output{i-1}.data, labels, layers{i}.num, 0, 1);
if nargout >= 2
    param_grad{i-1}.w = reshape(grad(1:length(params{i-1}.w(:))), size(params{i-1}.w));
    param_grad{i-1}.b = reshape(grad(end - length(params{i-1}.b(:)) + 1 : end), size(params{i-1}.b));
    param_grad{i-1}.w = param_grad{i-1}.w / batch_size;
    param_grad{i-1}.b = param_grad{i-1}.b /batch_size;
end

cp.cost = cost/batch_size;
cp.percent = percent;

if nargout >= 2
    % range: [l-1, 2]
    for i = l-1:-1:2
        switch layers{i}.type
            case 'CONV'
                output{i}.diff = input_od;
                [param_grad{i-1}, input_od] = conv_layer_backward(output{i}, output{i-1}, layers{i}, params{i-1});
            case 'POOLING'
                output{i}.diff = input_od;
                [input_od] = pooling_layer_backward(output{i}, output{i-1}, layers{i});
                param_grad{i-1}.w = [];
                param_grad{i-1}.b = [];
            case 'IP'
                output{i}.diff = input_od;
                [param_grad{i-1}, input_od] = inner_product_backward(output{i}, output{i-1}, layers{i}, params{i-1});
            case 'RELU'
                output{i}.diff = input_od;
                [input_od] = relu_backward(output{i}, output{i-1}, layers{i});
                param_grad{i-1}.w = [];
                param_grad{i-1}.b = [];
            case 'ELU'
                output{i}.diff = input_od;
                [input_od] = elu_backward(output{i}, output{i-1}, layers{i});
                param_grad{i-1}.w = [];
                param_grad{i-1}.b = [];
        end
        param_grad{i-1}.w = param_grad{i-1}.w / batch_size;
        param_grad{i-1}.b = param_grad{i-1}.b / batch_size;
    end
end

end
