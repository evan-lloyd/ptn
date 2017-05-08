function z = ttt(x, y, dims_x, dims_y)
%TTT Tensor-times-tensor for two htensors.
%
%  TTT(X,Y) computes the outer product of htensors X and Y.
% 
%  TTT(X,Y,XDIMS,YDIMS) computes the contracted product of tensors X
%  and Y in the modes specified by the row vectors XDIMS and
%  YDIMS. The sizes in the modes specified by XDIMS and YDIMS must
%  match, i.e. size(X,XDIMS) == size(Y,YDIMS).
% 
%  TTT(X,Y,DIMS) is equivalent to calling TTT(X,Y,DIMS,DIMS).
%
%  Additionally, the trees must fulfill the following requirements:
%
%  1. a) There is a node of X with a subtree comprising exactly the
%  dimensions in XDIMS, or in the complement of XDIMS. The same is
%  true for Y and YDIMS.
%
%  - OR -
%
%  1. b) All the dimensions of X are in XDIMS. There are two nodes in
%  Y, such that YDIMS is exactly the union of their subtrees or the
%  complements of their subtrees. The distance between these nodes is
%  exactly 3, i.e. there are two nodes between them. In the above,
%  X/XDIMS and Y/YDIMS may be swapped.
%
%  - AND -
%
%  2. The contracted parts of both dimension trees must have the same
%  structure.
%
%  In the case of complex tensors, we take the complex conjugate of X.
%
%  Examples
%  X = htenrandn([4,2,3]);
%  Y = htenrandn([3,4,2]);
%  Z = ttt(X,Y) %<-- outer product of X and Y
%  Z = ttt(X,X,1:3) %<-- inner product of X with itself
%  Z = ttt(X,Y,[1 2 3],[2 3 1]) %<-- inner product of X & Y
%  Z = ttt(X,Y,[1 3],[2 1]) %<-- product of X & Y along specified
%                           %dims
% 
%  See also TTM.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

% Check number of arguments
if(nargin <= 1)
  error('Requires at least 3 arguments.');
elseif(nargin == 2)
  dims_x = [];
end

if(~exist('dims_y', 'var'))
  dims_y = dims_x;
end

% Check x, y
if(~isa(x, 'htensor') || ~isa(y, 'htensor'))
  error('X and Y must be of class htensor.');
end

% Check dims_x, dims_y
if( ~isindexvector(dims_x) || numel(unique(dims_x)) < numel(dims_x) || ...
    ~isindexvector(dims_y) || numel(unique(dims_y)) < numel(dims_y) )
  error(['IND_X and IND_Y must be vectors of positive integers,' ...
	 ' without double entries.']);
end

% Check tensor sizes
sz_x = size(x);
sz_y = size(y);
if(~isequal(sz_x(dims_x), sz_y(dims_y)))
  error('Tensor dimensions must agree.');
end

% Find out what case this is, and the subtrees in question:
compl_dims_x = setdiff(1:ndims(x), dims_x);
compl_dims_y = setdiff(1:ndims(y), dims_y);

% Find subtree containing all dims_x or all compl_dims_x
ind_top_x = ...
    find(cellfun(@(s)(isequal(sort(s), sort(dims_x))), x.dims));
ind_top_x_compl = ...
    find(cellfun(@(s)(isequal(sort(s), sort(compl_dims_x))), ...
		 x.dims));

if(~isempty(ind_top_x))
  compl_x = false;
elseif(~isempty(ind_top_x_compl))
  compl_x = true;
  ind_top_x = ind_top_x_compl;
end

% Find subtree containing all dims_y or all compl_dims_y
ind_top_y = ...
    find(cellfun(@(s)(isequal(sort(s), sort(dims_y))), y.dims));
ind_top_y_compl = ...
    find(cellfun(@(s)(isequal(sort(s), sort(compl_dims_y))), ...
		 y.dims));

if(~isempty(ind_top_y))
  compl_y = false;
elseif(~isempty(ind_top_y_compl))
  compl_y = true;
  ind_top_y = ind_top_y_compl;
end

if(exist('compl_x', 'var') && exist('compl_y', 'var'))
  
  if(compl_x == false)
    x = change_root(x, ind_top_x, 'right');
  else
    x = change_root(x, ind_top_x, 'left');
  end
  
  if(ind_top_x == 1)
    if(isempty(dims_x))
      dims_x = 1;
    else
      dims_x = dims_x + 1;
    end
  end
  
  if(compl_y == false)
    y = change_root(y, ind_top_y, 'right');
  else
    y = change_root(y, ind_top_y, 'left');
  end
  
  if(ind_top_y == 1)
    if(isempty(dims_y))
      dims_y = 1;
    else
      dims_y = dims_y + 1;
    end
  end

  M = elim_matrix(x, y, dims_x, dims_y);
  
  x.U = cellfun(@conj, x.U, 'UniformOutput', false);  
  x.B = cellfun(@conj, x.B, 'UniformOutput', false);
  
  z = combine_trees(x, y, M);
  
elseif( numel(dims_x) == ndims(x) || numel(dims_y) == ndims(y) )
  
  if( numel(dims_x) == ndims(x) )
    z = x; x = y; y = z; clear z;
    dims_z = dims_x; dims_x = dims_y; dims_y = dims_z; clear dims_z;
    x = conj(x);
    y = conj(y);
  end
  
  dims_y_to_x(dims_y) = dims_x;
  
  for jj=1:y.nr_nodes
    dims_y_1 = y.dims{jj};
    dims_y_2 = setdiff(1:ndims(y), dims_y_1);
    
    dims_x_1       = dims_y_to_x(dims_y_1);
    compl_dims_x_1 = setdiff(1:ndims(x), dims_x_1);
    
    ind_top_x_1 = ...
	find(cellfun(@(s)(isequal(sort(s), sort(dims_x_1))), x.dims));
    ind_top_x_1_compl = ...
	find(cellfun(@(s)(isequal(sort(s), sort(compl_dims_x_1))), ...
		     x.dims));
    
    if(~isempty(ind_top_x_1))
      compl_x_1 = false;
    elseif(~isempty(ind_top_x_1_compl))
      compl_x_1 = true;
      ind_top_x_1 = ind_top_x_1_compl;
    end    
    
    dims_x_2       = dims_y_to_x(dims_y_2);
    compl_dims_x_2 = setdiff(1:ndims(x), dims_x_2);
    
    ind_top_x_2 = ...
	find(cellfun(@(s)(isequal(sort(s), sort(dims_x_2))), x.dims));
    ind_top_x_2_compl = ...
	find(cellfun(@(s)(isequal(sort(s), sort(compl_dims_x_2))), ...
		     x.dims));
    
    if(~isempty(ind_top_x_2))
      compl_x_2 = false;
    elseif(~isempty(ind_top_x_2_compl))
      compl_x_2 = true;
      ind_top_x_2 = ind_top_x_2_compl;
    end
    
    if(exist('compl_x_1', 'var') && exist('compl_x_2', 'var'))
      ind_split_y = jj;
      break;
    else
      clear compl_x_1 compl_x_2;
    end
    
  end
  
  if(~exist('ind_split_y', 'var'))
    error(['The contracted product cannot be computed using only 3rd' ...
	   ' order tensors.']);
  end
  
  % y:   y.B{1}         \      /
  %     /    \           B __ B 
  %   y_1    y_2        /      \
  %                    x_1      x_2
  %
  y_1 = change_root(y, ind_split_y, 'right');
  if(compl_x_1 == false)
    x_1 = change_root(x, ind_top_x_1, 'right');
  else
    x_1 = change_root(x, ind_top_x_1, 'left');
  end
  M1 = elim_matrix(x_1, y_1, dims_x_1, dims_y_1);
  
  y_2 = change_root(y_1, y_1.children(1, 1), 'right');
  if(compl_x_2 == false)
    x_2 = change_root(x, ind_top_x_2, 'right');
  else
    x_2 = change_root(x, ind_top_x_2, 'left');
  end
  M2 = elim_matrix(x_2, y_2, dims_x_2, dims_y_2);
  
  M_y = M1*M2.';
  
  node_x_1 = x_1.children(1, 1);
  if(all(ismember(dims_x_2, x_1.dims{x_1.children(node_x_1, 1)})) || ...
     all(ismember(compl_dims_x_2, x_1.dims{x_1.children(node_x_1, 1)})))
    lr_x_1 = 1;
  elseif(all(ismember(dims_x_2, x_1.dims{x_1.children(node_x_1, 2)})) || ...
	 all(ismember(compl_dims_x_2, x_1.dims{x_1.children(node_x_1, 2)})))
    lr_x_1 = 2;
  else
    error('This should not happen...')
  end
  
  node_x_2 = x_2.children(1, 1);
  if(all(ismember(dims_x_1, x_2.dims{x_2.children(node_x_2, 1)})) || ...
     all(ismember(compl_dims_x_1, x_2.dims{x_2.children(node_x_2, 1)})))
    lr_x_2 = 1;
  elseif(all(ismember(dims_x_1, x_2.dims{x_2.children(node_x_2, 2)})) || ...
	 all(ismember(compl_dims_x_1, x_2.dims{x_2.children(node_x_2, 2)})))
    lr_x_2 = 2;
  else
    error('This should not happen...')
  end
  
  M = ttt(x_1.B{node_x_1}, ttm(x_2.B{node_x_2}, M_y, 3), ...
	  [3 lr_x_1], [3 lr_x_2]);
  
  z = combine_trees(change_root(x_1, x_1.children(node_x_1, 3-lr_x_1), ...
				'left'), ...
		    change_root(x_2, x_1.children(node_x_2, 3-lr_x_2), ...
				'left'), ...
		    M);
else
  
  if( ~exist('compl_x', 'var') )
    error(['Argument DIMS_X must belong to a subtree, ' ...
	   'or to the complement of a subtree, as DIMS_Y does not' ...
	   ' cover all dimensions.']);
  end
  
  if( ~exist('compl_y', 'var') )
    error(['Argument DIMS_Y must belong to a subtree, ' ...
	   'or to the complement of a subtree, as DIMS_X does not' ...
	   ' cover all dimensions.']);
  end
  
end


function z = combine_trees(x, y, M)
% Combines the left subtree of x, the left subtree of y and the
% root matrix M to form a new htensor z:
%
% z :    M
%       / \
%  left_x  left_y
%

% Separate dimensions of x and y
y.dims = cellfun(@(s)(s+ndims(x)), y.dims, 'UniformOutput', false);

% Combine x and y in one big tree, with some unused nodes
offset_x = 1;
offset_y = x.nr_nodes + 1;

new_root_x = x.children(1, 1);
new_root_y = y.children(1, 1);

x.children(x.children ~= 0) = x.children(x.children ~= 0) + offset_x;
y.children(y.children ~= 0) = y.children(y.children ~= 0) + offset_y;

children = [offset_x + new_root_x, offset_y + new_root_y; x.children; y.children];

B{1} = M;
B(offset_x+(1:x.nr_nodes)) = x.B;
B(offset_y+(1:y.nr_nodes)) = y.B;

U(offset_x+(1:x.nr_nodes)) = x.U;
U(offset_y+(1:y.nr_nodes)) = y.U;

dims{1} = [x.dims{new_root_x}, y.dims{new_root_y}];
dims(offset_x+(1:x.nr_nodes)) = x.dims;
dims(offset_y+(1:y.nr_nodes)) = y.dims;

% Eliminate unused nodes
old2new = htensor.subtree(children, 1);
new2old(old2new) = 1:length(old2new);

children = children(old2new, :);
for ii=1:size(children, 1)
  if( any(children(ii, :) ~= 0) )
    children(ii, :) = new2old(children(ii, :));
  end
end

dims = dims(old2new);
U = U(old2new);
B = B(old2new);

new2old = sort(dims{1});
old2new(new2old) = 1:length(new2old);

dims = cellfun(@(s)(old2new(s)), dims, 'UniformOutput', ...
	       x.is_orthog & y.is_orthog);

z = htensor(children, dims, U, B, false);



function M = elim_matrix(x, y, dims_x, dims_y)
%
% Calculate the elimination matrix from the left/right subtrees of
% x and y, linked by dims_x and dims_y.
%
% Note that dims_x, dims_y must each cover the right or left
% subtree of x, y. This is not checked for.

% Calculate elimination matrix
M = cell(x.nr_nodes, 1);

dims_x_to_y(dims_x) = dims_y;
inds_x = htensor.subtree(x.children, x.children(1, 2));


% Start at leaves, move up the levels
for ii=inds_x(end:-1:1)
  
  dims_jj = dims_x_to_y(x.dims{ii});
  
  jj = find(cellfun(@(var)(isequal(sort(dims_jj), sort(var))), ...
		    y.dims));
  
  if(~isscalar(jj))
    error('Dimension trees of X and Y are incompatible.');
  end
  
  if(x.is_leaf(ii))
      % M_t = U1_t' * U2_t
      M{jj} = full(x.U{ii}'*y.U{jj});
  else
    jj_left  = y.children(jj, 1);
    jj_right = y.children(jj, 2);
    
    ii_left  = x.children(ii, 1);
    
    % M_t = B1_t' * (M_t1 kron M_t2) * B2_t
    B_ = ttm(y.B{jj}, { M{jj_left}, M{jj_right} }, [1 2]);
    
    if(isequal( sort(y.dims{jj_left}), ...
		sort(dims_x_to_y(x.dims{ii_left})) ))
      M{jj} = ttt(x.B{ii}, B_, [1 2]);
    elseif(isequal( sort(y.dims{jj_right}), ... 
		    sort(dims_x_to_y(x.dims{ii_left})) ))
      M{jj} = ttt(x.B{ii}, B_, [1 2], [2 1]);
    else
      error('This should not happen...')
    end
    
    % Save memory
    M{jj_left} = []; M{jj_right} = [];
  end
  
end

M = M{y.children(1, 2)};
