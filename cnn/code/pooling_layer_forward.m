% #############################################################################################
% #########  YOU SHOULD MAINTAIN THE RETURN TYPE AND SHAPE AS PROVIDED IN STARTER CODE   ######
% #############################################################################################

function [output] = pooling_layer_forward(input, layer)
% Pooling forward

% Args:
% input: a cell array contains output data and shape information
% layer: one cnn layer, defined in testLeNet.m

% Returns:
% output: a cell array contains output data and shape information

h_in = input.height;
w_in = input.width;
c = input.channel;
batch_size = input.batch_size;
k = layer.k;
pad = layer.pad;
stride = layer.stride;

h_out = (h_in + 2*pad - k) / stride + 1;
w_out = (w_in + 2*pad - k) / stride + 1;

output.height = h_out;
output.width = w_out;
output.channel = c;
output.batch_size = batch_size;

output.data = zeros([h_out * w_out * c, batch_size]);

input_n.height = h_in;
input_n.width = w_in;
input_n.channel = c;


% TODO: your implementation goes below this comment
% implementation begins

for n=1:batch_size
    input_n.data = input.data(:, n);
    col = im2col_conv_matlab(input_n, layer, h_out, w_out);
    col = reshape(col, k*k*c, h_out*w_out);
    pool = zeros(h_out*w_out,c);
    for i =1:c
        tmp = col(k*k*(i-1)+1:k*k*i,:);
        [max_,~] = max(tmp,[],1);
        pool(:,i) = max_;
    end
    output.data(:,n) = pool(:);
end
% 
% for i=1:batch_size
%     Thisbatch = input.data(:, i);
%     thisbatch = reshape(Thisbatch,h_in*w_in,c);
%     allchannel = zeros(h_out*w_out,c);
%     
%     for j=1:c
%         Thisim = thisbatch(:,j);
%         thisim = reshape(Thisim,[h_in,w_in]);
%         cal = 1;
%         pool = zeros(k*k,h_in*w_in/(k*k));
%         
%         for row=1:h_out
%             tworow = thisim((row-1)*k+1:row*k,:);
%             for col=1:w_out
%                 tmp = tworow(:,(col-1)*k+1:col*k);
%                 pool(:,cal) = tmp(:);
%                 cal = cal+1;
%             end      
%         end
%         [max_,~] = max(pool);
%         maxtmp = reshape(max_,h_out,w_out);
%         maxtmp = maxtmp';
%         allchannel(:,j) = maxtmp(:);
%     end
% 
%     output.data(:,i) = allchannel(:);
%     
% end

% implementation ends

assert(all(size(output.data) == [h_out * w_out * c, batch_size]), 'output.data does not have the right length');

end

