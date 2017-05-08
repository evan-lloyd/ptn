function y = squeeze(x, dims)
%SQUEEZE Remove singleton dimensions from an htensor.
%
%   Y = SQUEEZE(X) returns a tensor Y with the same elements as X, but
%   no singleton dimensions. A singleton dimension is a dimension DIM
%   such that size(X, DIM) == 1.
%
%   Y = SQUEEZE(X, DIMS) returns a tensor Y with the same elements as
%   X, with all the singleton dimensions in DIMS removed.
%
%   Exception: If all but one dimensions are singleton, the
%   returned htensor will have dimension n x 1, as ndims(x) >= 2
%   for an htensor.
%
%   Examples
%   squeeze( htenrandn([2,1,3]) ) %<-- returns a 2-by-3 tensor
%   squeeze( htenrandn([1 1]) )   %<-- returns a scalar
%
%   See also HTENSOR.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

% Check for case of scalar return value
if( all(size(x) == 1) )
  y = full(x);
  return;
end

if(nargin == 1)
  dims = 1:ndims(x);
elseif(~isindexvector(dims) || max(dims) > ndims(x) || ...
       ~numel(dims) == numel(unique(dims)) )
  error('DIMS must be a vector of integers between 1 and ndims(X).')
end

to_squeeze = false(1, ndims(x));
to_squeeze(dims) = true;

y.size      = size(x);
y.brother   = x.brother;
y.parent    = x.parent;
y.dim2ind   = x.dim2ind;
y.is_left   = x.is_left;
y.B         = x.B;
y.U         = x.U;
y.is_leaf   = x.is_leaf;
y.children  = x.children;
y.nr_nodes  = x.nr_nodes;
y.dims      = x.dims;
y.is_orthog = x.is_orthog;

% Remove singleton dimensions
while( length(find(y.size ~= -1)) > 2 )

  % Index of next leaf node to squeeze away
  ind = next_single_node(y, to_squeeze);
  
  % Stop if no singleton dimensions are left
  if(ind == -1)
    break;
  end
  
  % Index of brother and parent nodes
  ind_brother   = y.brother(ind);
  ind_par       = y.parent(ind);
  
  % Dimension corresponding to node ind
  d_ind = find(y.dim2ind == ind);
  
  % Apply U{ii} to parent tensor B, store resulting matrix in tmp
  if(y.is_left(ind))
    tmp = ttm(y.B{ind_par}, y.U{ind}, 1);
    tmp = reshape(tmp, [size(tmp, 2), size(tmp, 3)]);
  else
    tmp = ttm(y.B{ind_par}, y.U{ind}, 2);
    tmp = reshape(tmp, [size(tmp, 1), size(tmp, 3)]);
  end
  
  if(y.is_leaf(ind_brother))
    % Case brother node is also a leaf
    
    % Dimension corresponding to brother node
    d_brother = find(y.dim2ind == ind_brother);
    
    % Apply brother node to matrix
    y.U{ind_par} = y.U{ind_brother}*tmp;
    
    % Set previous B{ind_par} to empty tensor
    y.B{ind_par} = zeros([0 0 0]);
    
    % Update is_leaf
    y.is_leaf(ind_par) = true;
    
    % disconnect ind, ind_brother from dimension tree
    y.children(ind_par, :)  = [0 0];
    y.parent(ind)  = 0;
    y.parent(ind_brother) = 0;
    
    % Set dimension d_ind to -1
    y.size(d_ind) = -1;
    
    % Represent d_brother by ind_par
    y.dim2ind(d_brother) = ind_par;
    
  else
    % Case brother node has a subtree
    
    % Apply B{ind_brother} to B{ind_par}
    y.B{ind_par} = ttm(y.B{ind_brother}, tmp.', 3);
    
    % Nodes ind and ind_brother are not used anymore.
    % Change tree to directly connect ind_brother's
    % children to ind_par.
    y.children(ind_par, :) = y.children(ind_brother, :);
    y.parent(y.children(ind_brother, 1)) = ind_par;
    y.parent(y.children(ind_brother, 2)) = ind_par;   
    
    % Set dimension d_ind to -1
    y.size(d_ind) = -1;
    
  end
  
end


% Eliminate unused dimensions

% Indexes of dimensions that are still used
new2old_dims = find(y.size ~= -1);

% Other direction: new dimension index for all old dimensions
old2new_dims = zeros(size(y.size));
old2new_dims(new2old_dims) = 1:length(new2old_dims);

% Change dimension names in y.dims{ii}, and eliminate the unused ones.
for ii=1:y.nr_nodes
  y.dims{ii} = old2new_dims(y.dims{ii});
  y.dims{ii} = y.dims{ii}(y.dims{ii} ~= 0);
end

% Eliminate unused nodes

% Indexes of nodes that are still connected to the tree
new2old_nodes = htensor.subtree(y.children, 1);

% Other direction: new node index for all old nodes
old2new_nodes = zeros(y.nr_nodes, 1);
old2new_nodes(new2old_nodes) = 1:length(new2old_nodes);

% Eliminate unused nodes in y.children
y.children = y.children(new2old_nodes, :);
y.dims     = y.dims(new2old_nodes);
y.B        = y.B(new2old_nodes);
y.U        = y.U(new2old_nodes);

% Update node indexes in y.children
y.children(y.children ~= 0) = ...
    old2new_nodes(y.children(y.children ~= 0));

y = htensor(y.children, y.dims, y.U, y.B, y.is_orthog);

% In case d == 2 and size == [1 n]: permute
if(ndims(y) == 2 && y.size(1) == 1 && y.size(2) > 1)
  y = permute(y, [2 1]);
end


function ind = next_single_node(x, to_squeeze)
% Returns node indexes of all single nodes of x
% If two brother leafs have singleton dimensions, these nodes are
% returned first.

% All leaf nodes of singleton dimensions
single_nodes = x.dim2ind(x.size == 1 & to_squeeze);

% Parents of two leaf nodes with singleton dimensions
single_node_par = x.parent(single_nodes);
par_two_single_nodes = double_entries(single_node_par);

if(~isempty(par_two_single_nodes))
  % Return child of parent node, if there is one:
  ind = x.children(par_two_single_nodes(1), 1);
  
elseif(~isempty(single_nodes))
  % Otherwise, return any singleton node, if one exist
  ind = single_nodes(1);
else
  % No singleton dimensions left
  ind = -1;
end


function v = double_entries(v)
% Returns only the entries that appear more than once in vector v.

v = sort(v);

[tmp, idx1] = unique(v, 'first');
[tmp, idx2] = unique(v, 'last');

v = v(idx1(idx1~=idx2));
