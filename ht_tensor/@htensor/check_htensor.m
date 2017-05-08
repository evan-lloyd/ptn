function check_htensor(x)
%CHECK_HTENSOR Check internal consistency of htensor.
%
%   CHECK_HTENSOR(X) returns an error message if there is an
%   inconsistency in the fields of X.
%
%   This function is used by the constructor HTENSOR(CHILDREN, DIMS,
%   U, B, [IS_ORTHOG]), and can be used for debugging purposes.
%
%   See also HTENSOR.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

% Check first argument children
if(~isnumeric(x.children) || size(x.children, 2) ~= 2 || ...
   any(any(~ismember(x.children, 0:size(x.children, 1)))) )
  error(['CHILDREN must be an N x 2 array of integers between' ...
	 ' 0 and N, defining a binary tree.']);
end

if(~isequal(sort(htensor.subtree(x.children, 1)), 1:x.nr_nodes))
  error(['Some indexes of CHILDREN are not part of the tree, or' ...
	 ' the root node does not have index 1.']);
end

% Check that children come after their parents in the node
% indexing, and that all leaf nodes are [0 0].
ind1 = find(x.children(:, 1) ~= 0);
ind2 = find(x.children(:, 2) ~= 0);
if(~isequal(ind1, ind2))
  error('Each node should have either no or two children.');
end
tmp = x.children - (1:x.nr_nodes)'*ones(1, 2);
if(tmp(ind1, :) <= 0)
  error('Child nodes must have higher indexes than their parent node.');
end

% Check dims
if(~isa(x.dims, 'cell') || ...
   any(cellfun(@(var)(~isnumeric(var)), x.dims)) || ...
   any(cellfun('size', x.dims, 1) ~= 1))
  error('DIMS must be a cell array of row vectors.')
end

if( length(x.dims) ~= size(x.children, 1) )
  error('Inconsistent number of nodes for CHILDREN and DIMS.');
end

% Check root value of dims
if(~isequal(sort(x.dims{1}), 1:ndims(x)))
  error('Invalid DIMS, dimensions must be part of 1: ndims(x).');
end

% Check internal consistency of dims
for ii=1:length(x.dims)
  if(x.is_leaf(ii))
    if(numel(x.dims{ii}) ~= 1)
      error('Leaf nodes should only contain one dimension.')
    end
  else
    ii_left  = x.children(ii, 1);
    ii_right = x.children(ii, 2);
    if(~isequal(x.dims{ii}, [x.dims{ii_left}, x.dims{ii_right}]))
      error(['Inconsistency in DIMS, parent node must contain' ...
	     ' dimensions of child nodes, left to right.']);
    end
  end
end

% Check data types of U and B
if(~isa(x.U, 'cell') || ...
   ~isa(x.B, 'cell') || ...
   ~all(cellfun(@(x)(isa(x, 'numeric')), x.U)) || ...
   ~all(cellfun(@(x)(isa(x, 'numeric')), x.B)) )
  error('U and B must be cell arrays of MATLAB arrays.');
end

if( size(x.U, 1) ~= 1 || size(x.B, 1) ~= 1 )
  error(['U and B must be row cell arrays, i.e. of size' ...
	 ' 1xnr_nodes.']);
end

% Check ndims of U and B
if( ~all(cellfun('ndims', x.U) == 2) )
  error('U{ii} must be a matrix.');
end

if( ~all(ismember(cellfun('ndims', x.B), [2 3])) )
  error('B{ii} must be an array with ndims = 3 or ndims = 2.');
end

% Check sizes of U and B
for ii=2:length(x.dims)
  ii_par = x.parent(ii);
  if(x.is_left(ii))
    left_right = 1;
  else
    left_right = 2;
  end
  
  if(x.is_leaf(ii))
    
    if(size(x.U{ii}, 2) ~= size(x.B{ii_par}, left_right))
      error('Inconsistent dimensions of U and B.');
    end
  else
    if(size(x.B{ii}, 3) ~= size(x.B{ii_par}, left_right))
      error('Inconsistent dimensions of U and B.');
    end      
    % Check that no rank is 0.
    if(any(size(x.B{ii}) == 0))
      error('All ranks must be 1 or bigger.');
    end
  end
  
end

% Check that root node has rank 1.
if(size(x.B{1}, 3) ~= 1)
  error('The root node must have rank 1.');
end

% Check is_orthog
if( ~isa(x.is_orthog, 'logical') || ~isscalar(x.is_orthog) )
  error('IS_ORTHOG must be a (scalar) logical.');
end
