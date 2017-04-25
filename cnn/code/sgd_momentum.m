% #############################################################################################
% #########  YOU SHOULD MAINTAIN THE RETURN TYPE AND SHAPE AS PROVIDED IN STARTER CODE   ######
% #############################################################################################

function [params, param_winc] = sgd_momentum(w_rate, b_rate, mu, weight_decay, params, param_winc, param_grad)
% update the parameters with sgd with momentum

%% function input
% w_rate (scalar): learning rate for w at current step
% b_rate (scalar): learning rate for b at current step
% mu (scalar): momentum
% weight_decay (scalar): weigth decay of w
% params (cell array): original weight parameters
% param_winc (cell array): buffer to store history gradient accumulation
% param_grad (cell array): gradient of parameter

%% function output
% params (cell array): updated parameters
% param_winc (cell array): updated buffer

% TODO: your implementation goes below this comment
% implementation begins

% implementation ends
for i=1:length(params)
    if ~isempty(param_grad{i}.w)
        gradw = param_grad{i}.w + weight_decay * params{i}.w;
        gradb = param_grad{i}.b;
        thetaw = mu * param_winc{i}.w + w_rate * gradw;
        thetab = mu * param_winc{i}.b + b_rate * gradb;
        param_winc{i}.w = thetaw;
        param_winc{i}.b = thetab;
        params{i}.w = params{i}.w - param_winc{i}.w;
        params{i}.b = params{i}.b - param_winc{i}.b;
    end
end


%     gradw = param_grad[i + 1]['w'] + decay * params[i + 1]['w']
%     gradb = param_grad[i + 1]['b'] + params[i + 1]['b']
%     thetaw = param_winc[i + 1]['w']
%     thetab = param_winc[i + 1]['b']
%     thetaw = mu * thetaw + w_rate * gradw
%     thetab = mu * thetab + b_rate * gradb
%     params_[i + 1]['w'] = params[i + 1]['w'] - thetaw
%     params_[i + 1]['b'] = params[i + 1]['b'] - thetab
%     param_winc_[i + 1]['w'] = thetaw
%     param_winc_[i + 1]['b'] = thetab
    
    
assert(all(size(params) == size(param_grad)), 'params_ does not have the right length');
assert(all(size(param_winc) == size(param_grad)), 'param_winc_ does not have the right length');

end
