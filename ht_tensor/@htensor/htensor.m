classdef htensor
%HTENSOR - Hierarchical Tucker tensor.
%
%   A MATLAB class representing a Hierarchical Tucker tensor, allowing
%   its construction and standard operations with such
%   tensors. Functions are provided for approximating a full tensor
%   in H-Tucker format, converting a CP tensor to the H-Tucker
%   format and truncating a tensor in H-Tucker format to another
%   tensor in H-Tucker format with smaller hierarchical ranks.
%
%   See also HTENSOR/HTENSOR

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

properties( SetAccess = private, GetAccess = public )

children   % Structure of the dimension tree.
dims       % Dimensions each node represents.
U          % Matrices in the leaf nodes.
B          % Transfer tensors in non-leaf nodes.
is_orthog  % Indicates whether the tensor is orthogonalized.

end

% Dependent properties
properties( Dependent = true, SetAccess = private, GetAccess = public )

parent     % Parent of each node.
nr_nodes   % Number of nodes in tree.
is_leaf    % Indicates whether node is a leaf.
lvl        % Level (distance from root) of each node.
dim2ind    % Index of leaf node for each dimension.
is_left    % Indicates if the node is a left child.
is_right   % Indicates if the node is a right child.
brother    % Brother of the node.

end

% Get methods for dependent properties
methods

  function nr_nodes = get.nr_nodes(x)
    nr_nodes = size(x.children, 1);
  end
   
  function is_leaf = get.is_leaf(x)
    is_leaf = all(x.children == 0, 2)';
  end

  function lvl = get.lvl(x)
    lvl = -ones(1, x.nr_nodes);
    lvl(1) = 0;
    ind = htensor.subtree(x.children, 1);
    for ii=ind(2:end)
      lvl(ii) = lvl(x.parent(ii)) + 1;
    end
  end
   
  function parent = get.parent(x)
    parent = zeros(1, x.nr_nodes);
    for ii=find(~x.is_leaf)   
      parent(x.children(ii, 1)) = ii;
      parent(x.children(ii, 2)) = ii;
    end
  end
     
  function dim2ind = get.dim2ind(x)
    ind = find(cellfun('length', x.dims) == 1);
    leaf_dims = cell2mat(x.dims(ind));
    [tmp, idx] = sort(leaf_dims);        
    dim2ind = ind(idx);
  end
  
  function is_left = get.is_left(x)
    left_nodes = x.children(x.children(:, 1) ~= 0, 1);
    is_left = false(x.nr_nodes, 1);
    is_left(left_nodes) = true;
  end
   
  function is_right = get.is_right(x)
    right_nodes = x.children(x.children(:, 2) ~= 0, 2);
    is_right = false(x.nr_nodes, 1);
    is_right(right_nodes) = true;
  end
  
  function brother = get.brother(x)
    brother = zeros(x.nr_nodes, 1);
    ind = find(x.parent ~= 0);
    child_nodes(ind, :) = x.children(x.parent(ind), :);
    brother(x.is_left) = child_nodes(x.is_left, 2);
    brother(x.is_right) = child_nodes(x.is_right, 1);
  end
end
    

methods( Access = public )

function x = htensor(varargin)
%HTENSOR Construct a hierarchical tensor.
%
%   X = HTENSOR() creates a 1x1 htensor with entry zero.
%
%   X = HTENSOR(SZ, [TREE_TYPE]) creates a zero htensor with SZ =
%   size(A). The string TREE_TYPE controls the shape of the dimension
%   tree, default is a balanced tree.
%
%   X = HTENSOR(CP, WEIGHTS, [TREE_TYPE]) converts a tensor in CP
%   decomposition into an htensor. CP may be a cell array, or a
%   Tensor Toolbox ktensor. When CP is a cell array, the vector
%   WEIGHTS can be used to assign a weight to each rank-one
%   summand. TREE_TYPE controls the shape of the dimension tree,
%   default is a balanced tree.
%
%   X = HTENSOR(H) copies the htensor H.
%
%   X = HTENSOR(CHILDREN, DIMS, U, B, [IS_ORTHOG]) constructs an
%   htensor by arguments.
%
%   For TREE_TYPE options, see HTENSOR.DEFINE_TREE.
%
%   See also HTENRAND, TRUNCATE, (Tensor Toolbox KTENSOR).

% Default constructor
if(nargin == 0)
  
  x = htensor([1 1]);
  return;
  
elseif(nargin <= 3)
  
  % Copy constructor
  if(isa(varargin{1}, 'htensor'))
    x = varargin{1};
    
    % Conversion constructor from tensor_toolbox class ktensor
  elseif(isa(varargin{1}, 'ktensor') || isa(varargin{1}, 'cell'))
    
    % Check for empty cell array
    if(isempty(varargin{1}))
      error(['First argument CP cannot be an empty cell array,' ...
	     ' dimension of tensor in CP decomposition must be' ...
	     ' at least 1.']);
    end
    
    % Initialize variables
    if(isa(varargin{1}, 'ktensor'))
      kt = varargin{1};
      d = ndims(kt);
      cpU = kt.U;
      lambda = kt.lambda;
    else
      d = length(varargin{1});
      cpU = varargin{1};
      if(nargin >= 2 && isnumeric(varargin{2}) && ...
	 isvector(varargin{2}))
	lambda = varargin{2};
      else
	lambda = ones(size(cpU{1}, 2), 1);
      end
    end
    
    % Set minimum number of dimensions to 2
    if(d == 1)
      d = 2;
      cpU{2} = ones(1, length(lambda));
    end
    
    % Check input size and types
    rank = length(lambda);
    
    if(~all(cellfun(@(x)(isa(x, 'numeric')), cpU)))
      error('All elements of cell array CP must be numeric.')
    end
    
    if( any(cellfun('size', cpU, 2) ~= rank) || ...
	any(cellfun('size', cpU, 1) == 0) )
      error(['All arrays in CP must have the same number of columns, ' ...
	     'and must have at least one row.']);
    end
    
    % Initialize dimension tree
    if(nargin <= 2)
      [x.children, x.dims] = htensor.define_tree(1:d);
    else
      [x.children, x.dims] = htensor.define_tree(1:d, varargin{3});
    end
    
    % Construct cell arrays U and B
    x.U = cell(1, size(x.children, 1));
    x.B = cell(1, size(x.children, 1));
    
    for ii=2:size(x.children, 1)
      if(length(x.dims{ii}) == 1)
    	x.U{ii} = cpU{x.dims{ii}};
      else
	x.B{ii} = diag3d(ones(rank, 1));
      end
    end
    x.B{1} = diag(lambda);
    
    x.is_orthog = false;
    
    % zero-initialization
  elseif( isvector(varargin{1}) )
    x_size = varargin{1};
    
    % Make sure x_size has at least two entries, creating singleton
    % dimensions if necessary:
    x_size(end+1:2) = 1;
    
    % Check that all elements of x_size are positive integers
    if( any(ceil(x_size) ~= floor(x_size)) || any(x_size < 0) )
      error('The vector SZ can only contain positive integers.');
    end
    
    % Initialize dimension trees
    if(nargin==1)
      [x.children, x.dims] = htensor.define_tree(1:length(x_size));
    else
      [x.children, x.dims] = htensor.define_tree(1:length(x_size), ...
						 varargin{2});
    end
    
    % Fill U and B cell arrays
    x.U = cell(1, size(x.children, 1));
    x.B = cell(1, size(x.children, 1));
    
    for ii=1:size(x.children, 1)
      if(length(x.dims{ii}) == 1)
	x.U{ii} = zeros(x_size(x.dims{ii}), 1);
	x.B{ii} = [];
      else
	x.U{ii} = [];
        x.B{ii} = 1;
      end
    end
    
    x.is_orthog = false;
    
  else
    error('Invalid arguments.')
  end

elseif(nargin >= 4)
  % Insert data
  x.children = varargin{1};
  x.dims     = varargin{2};
  x.U        = varargin{3};
  x.B        = varargin{4};
  
  if(nargin >= 5)
    x.is_orthog   = varargin{5};
  else
    x.is_orthog = false;
  end
  
  % Check resulting htensor  
  check_htensor(x);
  
  % Ensure that U and B have correct length
  x.U(end+1:size(x.children, 1)) = {[]};
  x.B(end+1:size(x.children, 1)) = {[]};
  
end

end

  AB = apply_mat_to_mat(A, B, p)
  Ax = apply_mat_to_vec(A, x)
  x = conj(x)
  check_htensor(x)
  ctranspose(x)
  disp(x, name, v)
  disp_tree(x, name, v)
  display(x)
  [dofs, dofsB] = dofs(x)
  [z, z_, err, sv] = elem_mult(x, y, opts)
  e = end(x, k, n)
  comp = equal_dimtree(x1, x2)
  y = full(x)
  y = full_block(x, index)
  x = full_leafs(x)
  G = gramians(x)
  G = gramians_nonorthog(x)
  s = innerprod(x1, x2)
  x = minus(x1, x2)
  x = mrdivide(x, a)
  x = mtimes(a, b)
  M = mttkrp(x, U, n)
  d = ndims(x)
  nrm = norm(x)
  nrm = norm_diff(x, x_full, max_numel)
  x = orthog(x)
  x = permute(x, order)
  plot_sv(x, opts)
  x = plus(x1, x2)
  y = power(x, p)
  r = rank(x, idx)
  sz = size(x, idx)
  sv = singular_values(x, opts)
  x = sparse_leaves(x)
  spy(x, opts)
  y = squeeze(x, dims)
  x = subsasgn(x, s, v)
  out = subsref(x, s)
  transpose(x)
  [x, err, sv] = truncate_std(x, opts)
  [x, err, sv] = truncate_nonorthog(x, opts)
  x_tucker = ttensor(x)
  x = ttm(x, A, varargin)
  x = ttv(x, v, dims)
  z = ttt(x, y, dims_x, dims_y)
  x = uminus(x)
  x = uplus(x)
  x = change_root(x, ind, lr_subtree)
  U = nvecs(x, i, r)
  z = times(x, y)
  
end

methods( Static, Access = public )

  ind = subtree(children, ii, count)
  [children, dims] = define_tree(dims, opts)
  [U, s] = left_svd_gramian(G)
  [U, s] = left_svd_qr(A)
  [k, err, success] = trunc_rank(s, opts)
  G = gramians_sum(x_cell)
  G = gramians_cp(cp, weights, tree_type)

  [ht, err, sv] = truncate_ltr(x, opts)
  [ht, err, sv] = truncate_rtl(x, opts)
  [x, err, sv] = truncate_sum(x_cell, opts)
  [x, err, sv] = truncate_cp(cp, opts, weights)
     
end

end

