% ########################################################################
% #######     DO NOT MODIFY, READ ONLY IF YOU ARE INTERESTED       #######
% ########################################################################

function im = col2im_conv(col, input, layer, h_out, w_out)
% Convert image to columns

% Args:
% col: shape = (k*k, c, h_out*w_out)
% input: a cell array contains input data and shape information
% layer: one cnn layer, defined in testLeNet.m
% h_out: output height
% w_out: output width

% Returns:
% im: shape = (h_in, w_in, c)

im = col2im_conv_matlab(col, input, layer, h_out, w_out);
end