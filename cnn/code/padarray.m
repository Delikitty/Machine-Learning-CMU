% ########################################################################
% #######     DO NOT MODIFY, READ ONLY IF YOU ARE INTERESTED       #######
% ########################################################################

function B = padarray(A, padsize, padval)
 
  padsize = padsize(:).';
 
  if (~any (padsize))
    B = A;
    return
  end
 
  fancy_pad = false;
 
  B_ndims = max ([numel(padsize) ndims(A)]);
  A_size  = size (A);
  P_size  = padsize;
  A_size(end+1:B_ndims) = 1;  % add singleton dimensions
  P_size(end+1:B_ndims) = 0;  % assume zero for missing dimensions
 
  pre_pad_size = P_size;
  B_size = A_size + pre_pad_size + P_size;
 
  %% insert input matrix into output matrix
  A_idx = cell (B_ndims, 1);
  for dim = 1:B_ndims
    A_idx{dim} = (pre_pad_size(dim) +1):(pre_pad_size(dim) + A_size(dim));
  end
  B = repmat (cast (padval, class (A)), B_size);
  B(A_idx{:}) = A;
 
end
