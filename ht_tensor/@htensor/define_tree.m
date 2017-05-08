function [children, dims] = define_tree(dims, tree_type)
%DEFINE_TREE Define a dimension tree.
%
%   DEFINE_TREE(1:d) returns two arrays defining a balanced
%   dimension tree for an htensor of order d.
%
%   DEFINE_TREE(I) returns two arrays defining a balanced dimension
%   tree for an htensor of order numel(I). The sequence I must
%   contain a consecutive number of integers, i.e. sort(I) == 1:numel(I).
%
%   DEFINE_TREE(I, TREE_TYPE) returns two arrays defining a dimension
%   tree, with a structure depending on argument TREE_TYPE:
%
%   TREE_TYPE = 'first_separate': The left child of the root node
%                 is a leaf, the right subtree is balanced.
%   TREE_TYPE = 'TT': Corresponds to the Tensor Train format
%
%   The outputs:
%   CHILDREN, a nr_nodes x 2 array, specifying the indexes of each
%   node's two child nodes. Leaf nodes point to index 0 for both
%   child nodes.
%
%   DIMS is a 1 x nr_nodes cell array. Each entry specifies the
%   modes associated with the corresponding node.
%
%   See also: HTENSOR, CONSTR_TREE.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

if(nargin < 1)
  error('Requires at least 1 argument.')
elseif( ~isindexvector(dims) || numel(unique(dims)) ~= numel(dims) )
  error('First argument must be an index vector without double entries.')
end

if(nargin == 1)
  tree_type = '';
elseif(~ischar(tree_type))
  error('TREE_TYPE must be a char array.')
end

d = length(dims);

% Check input dims
if( any(sort(dims) ~= 1:d) )
  error('All dimensions from 1 to d must be used in DIMS.')
end

children = zeros(2*d-1, 2);
dims = {dims};

nr_nodes = 1;
ii = 1;

while(ii <= nr_nodes)
  
  if(length(dims{ii}) == 1)
    children(ii, :) = [0, 0];
  else
    ii_left  = nr_nodes + 1;
    ii_right = nr_nodes + 2;
    nr_nodes = nr_nodes + 2;
    
    children(ii, :) = [ii_left, ii_right];
    
    if(nargin == 2 && strcmp(tree_type, 'first_separate') && ii==1)
      dims{ii_left } = dims{ii}(1);
      dims{ii_right} = dims{ii}(2:end);      
    elseif(nargin == 2 && strcmp(tree_type, 'TT'))
      dims{ii_left } = dims{ii}(1);
      dims{ii_right} = dims{ii}(2:end);      
    else
      dims{ii_left } = dims{ii}(1:floor(end/2));
      dims{ii_right} = dims{ii}(floor(end/2)+1:end);
    end
  end
  
  ii = ii + 1;
end
