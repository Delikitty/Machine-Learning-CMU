% ########################################################################
% #######     DO NOT MODIFY, READ ONLY IF YOU ARE INTERESTED       #######
% ########################################################################

function col = im2col_conv(input_n, layer, h_out, w_out)
% Convert columns to image

% Args:
% input_n: input data, shape=(h_in*w_in*c, )
% layer: one cnn layer, defined in testLeNet.m
% h_out: output height
% w_out: output width

% Returns:
% col: shape=(k*k*c, h_out*w_out)

col = im2col_conv_matlab(input_n, layer, h_out, w_out);
end