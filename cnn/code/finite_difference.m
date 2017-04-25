% ########################################################################
% #######     DO NOT MODIFY, READ ONLY IF YOU ARE INTERESTED       #######
% ########################################################################

function [input_od_approx] = finite_difference(output, input, h)
% Args:
% input: input layer of our CNN, cell array which contains 'data'
% output: output layer of our CNN, cell array which contains 'data' and 'diff'
% h (scalar): finite difference

% Output:
% input_od_approx (size of input.data): approximate gradient using finite difference w.r.t input data

    x_plus_h = input;
    x_plus_h.data = x_plus_h.data + h;
    layer = 0;
    fx_plus_h = relu_forward(x_plus_h, layer);
    output.data;
    fx_plus_h.data;
    input_od_approx = ((fx_plus_h.data - output.data)/h).*output.diff;
end
