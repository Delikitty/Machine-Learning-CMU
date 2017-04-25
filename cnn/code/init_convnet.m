% ########################################################################
% #######     DO NOT MODIFY, READ ONLY IF YOU ARE INTERESTED       #######
% ########################################################################

function params = init_convnet(layers)
% Initialize parameters of each layer in LeNet

% Args:
% layers: a cell array that defines LeNet

% Returns:
% params: a cell array stores initialized parameters

h = layers{1}.height;
w = layers{1}.width;
c = layers{1}.channel;

% starts with second layer, first layer is data layer
for i = 2:length(layers)
    switch layers{i}.type
        case 'CONV'
            scale = sqrt(3/(h*w*c));
            params{i-1}.w = 2*scale*rand(layers{i}.k*layers{i}.k*c/layers{i}.group, layers{i}.num) - scale;
            params{i-1}.b = zeros(1, layers{i}.num);
            % update h, w and c, used in next layer
            h = (h + 2*layers{i}.pad - layers{i}.k) / layers{i}.stride + 1;
            w = (w + 2*layers{i}.pad - layers{i}.k) / layers{i}.stride + 1;
            c = layers{i}.num;
        case 'POOLING'
            h = (h - layers{i}.k) / layers{i}.stride + 1;
            w = (w - layers{i}.k) / layers{i}.stride + 1;
            params{i-1}.w = [];
            params{i-1}.b = [];
        case 'IP'
            switch layers{i}.init_type
                case 'gaussian'
                    scale = sqrt(3/(h*w*c));
                    params{i-1}.w = scale*randn(h*w*c, layers{i}.num);
                    params{i-1}.b = zeros(1, layers{i}.num);
                case 'uniform'
                    scale = sqrt(3/(h*w*c));
                    params{i-1}.w = 2*scale*rand(h*w*c, layers{i}.num) - scale;
                    params{i-1}.b = zeros(1, layers{i}.num);
            end
            h = 1;
            w = 1;
            c = layers{i}.num;
        case 'RELU'
            params{i-1}.w = [];
            params{i-1}.b = [];
        case 'ELU'
            params{i-1}.w = [];
            params{i-1}.b = [];
        case 'LOSS'
            scale = sqrt(3/(h*w*c));
            % last layer is K-1
            params{i-1}.w = 2*scale*rand(h*w*c, layers{i}.num - 1) - scale;
            params{i-1}.w = params{i-1}.w';
            params{i-1}.b = zeros(1, layers{i}.num - 1);
            params{i-1}.b = params{i-1}.b';
            h = 1;
            w = 1;
            c = layers{i}.num;
    end
end
end
