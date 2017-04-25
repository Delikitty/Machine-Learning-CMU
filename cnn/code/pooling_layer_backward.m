% #############################################################################################
% #########  YOU SHOULD MAINTAIN THE RETURN TYPE AND SHAPE AS PROVIDED IN STARTER CODE   ######
% #############################################################################################

function [input_od] = pooling_layer_backward(output, input, layer)
% Pooling backward

% Args:
% output: a cell array contains output data and shape information
% input: a cell array contains input data and shape information
% layer: one cnn layer, defined in testLeNet.m

% Returns:
% input_od: gradients w.r.t input data

h_in = input.height;
w_in = input.width;
h_out = output.height;
w_out = output.width;
c = input.channel;
batch_size = input.batch_size;
k = layer.k;

input_od = zeros(size(input.data));

% TODO: your implementation goes below this comment
% implementation begins

input_n.height = h_in;
input_n.width = w_in;
input_n.channel = c;
% 
for n=1:batch_size
    input_n.data = input.data(:, n);
    col = im2col_conv_matlab(input_n, layer, h_out, w_out);
    col = reshape(col, k*k*c, h_out*w_out);
    % pool = zeros(h_out*w_out,c);
    % allchannel = zeros(h_in*w_in,c);
    diff_batch = reshape(output.diff(:,n),[h_out*w_out,c]);
    for i =1:c
        diff_channel = diff_batch(:,i);
        tmp = col(k*k*(i-1)+1:k*k*i,:);
        [~,idx] = max(tmp,[],1);
        pool_zero = zeros(size(tmp));
        coordinate = [idx(:)';1:length(idx)];
        for number=1:length(idx)
            pool_zero(coordinate(1,number),coordinate(2,number)) = diff_channel(number);   
        end
        col(k*k*(i-1)+1:k*k*i,:) = pool_zero;
%         putback = zeros(h_in,w_in);
%         count = 1;
%         for back=1:w_out
%             part = pool_zero(:,w_out*(back-1)+1:w_out*(back-1)+w_out);
%             one = part(1:k,:);
%             putback(count,:) = reshape(one,[h_in,1]);
%             count = count+1;
%             two = part(k+1:k*k,:);
%             putback(count,:) = reshape(two,[h_in,1]);
%             count = count+1; 
%         end
%        allchannel(:,i) = putback(:);

    end
    allchannel = col2im_conv_matlab(col, input, layer, h_out, w_out);
    input_od(:,n) = allchannel(:);
end


% for i=1:batch_size
%     Thisbatch = input.data(:, i);
%     thisbatch = reshape(Thisbatch,h_in*w_in,c);
%     diff_batch = reshape(output.diff(:,i),[h_out*w_out,c])';
%     allchannel = zeros(h_in*w_in,c);
%     for j=1:c
%         diff_channel = diff_batch(j,:);
%         Thisim = thisbatch(:,j);
%         thisim = reshape(Thisim,[h_in,w_in])';
%         cal = 1;
%         pool = zeros(k*k,h_in*w_in/(k*k));
%         pool_zero = zeros(k*k,h_in*w_in/(k*k));
%         count = 1;
%         % use to put back the diffs and zeros
%         for row=1:h_out
%             tworow = thisim((row-1)*k+1:row*k,:);
%             for col=1:w_out
%                 tmp = tworow(:,(col-1)*k+1:col*k);
%                 pool(:,cal) = tmp(:);
%                 cal = cal+1;
%             end      
%         end
%         [~,idx] = max(pool);
%         idxtmp = reshape(idx,h_out,w_out);
%         idxtmp = idxtmp';
%         coordinate = [idxtmp(:)';1:length(idx)];
%         for number=1:length(idx)
%             pool_zero(coordinate(1,number),coordinate(2,number)) = diff_channel(number);  
%         end
%         % reshape this back to h_in * w_in
%         putback = zeros(h_in,w_in);
%         % is h_in*w_in matrix with a gradient in each patch
%         for back=1:w_out
%             part = pool_zero(:,w_out*(back-1)+1:w_out*(back-1)+w_out);
%             one = part(1:k,:);
%             putback(count,:) = reshape(one,[h_in,1]);
%             count = count+1;
%             two = part(k+1:k*k,:);
%             putback(count,:) = reshape(two,[h_in,1]);
%             count = count+1; 
%         end
%         allchannel(:,j) = putback(:);
%     end
%     % now use is a 4*144*20 matrix
%     input_od(:,i) = allchannel(:);
%     
% end
% implementation ends

assert(all(size(input.data) == size(input_od)), 'input_od does not have the right length');

end
