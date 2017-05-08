function nrm = norm_diff(x, x_full, max_numel)
%NORM_DIFF Norm of difference between htensor and full tensor.
%
%   NORM_DIFF(X, X_FULL, [MAX_NUMEL]) calculates the tensor norm ||X -
%   X_FULL|| for htensor X and (dense) tensor x_full, in a
%   storage-preserving manner: The full version of htensor x is
%   constructed consecutively, in blocks. The maximal size of each
%   block can be controlled by the optional parameter MAX_NUMEL
%   (default value 1e5 ~ 781KB).
%
%   See also HTENSOR, HTENSOR/NORM.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

if(nargin < 2)
  error('Requires at least 2 arguments.');
end

if(~isa(x, 'htensor'))
  error('First argument must be of class htensor.');
end

if(~isnumeric(x_full))
  error('Second argument must be a (multidimensional) array.');
end

% Set default value for max_numel
if(nargin == 2)
  max_numel = 1e5;
elseif(~isnumeric(max_numel) || ~isscalar(max_numel))
  error('MAX_NUMEL must be a scalar value.')
end

if(prod(size(x)) <= max_numel)
  full_x = full(x);
  nrm = norm(full_x(:) - x_full(:));
  return;
end

% Recursively construct blocks of the tensor x:

% index{ii} is a dx2-matrix defining a block of x
index{1} = [ones(ndims(x_full), 1), size(x_full)'];

% size_index{ii} is the size of a block of x, in each dimension
size_index{1} = index{1}(:, 2) - index{1}(:, 1) + 1;

% Find blocks that need to be reduced
blocks_to_reduce = find(cellfun(@prod, size_index) > max_numel);

while( ~isempty(blocks_to_reduce) )
  
  % Go through all blocks that need to be reduced
  for ii=blocks_to_reduce
    
    % find largest dimension of the block
    [red_size, red_ind] = max(size_index{ii});
    mid_point = floor((index{ii}(red_ind, 1) + index{ii}(red_ind, 2))/2);
    
    % Cut block up along largest dimension
    index{end+1} = index{ii};
    index{ii }(red_ind, 2) = mid_point;
    index{end}(red_ind, 1) = mid_point+1;
    
    % Update size of the block in each dimension
    size_index{ii }   = index{ii }(:, 2) - index{ii }(:, 1) + 1;
    size_index{end+1} = index{end}(:, 2) - index{end}(:, 1) + 1;
    
  end
  
  % Find blocks that still need to be reduced
  blocks_to_reduce = find(cellfun(@prod, size_index) > max_numel);
  
end

% initialize square of norm
nrm_sqr = 0;

% loop through all blocks, and add up their tensor norms
for ii=1:length(index)
  
  % Calculate dense tensor for block ii of htensor x
  x_local = full_block(x, index{ii});
  
  % Calculate corresponding block of dense tensor x_full
  eval_string = sprintf('%d:%d, ', index{ii}');
  eval_string = ['x_full(', eval_string(1:end-2), ');'];
  x_full_local = eval(eval_string);
  
  % add square of tensor norm
  nrm_sqr = nrm_sqr + norm(x_local(:) - x_full_local(:))^2;
  
end

nrm = sqrt(nrm_sqr);
