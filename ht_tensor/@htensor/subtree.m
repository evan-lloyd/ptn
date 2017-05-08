function sub_ind = subtree(children, ii)
%SUBTREE Returns all nodes in the subtree of a node.
%
%   T = SUBTREE(CHILDREN, I) recursively traverses the subtree at node
%   I of the tree defined by CHILDREN, and returns the indexes of all
%   nodes found. The resulting vector contains the indexes of all
%   descendant nodes of node ii.
%
%   See also: HTENSOR, DEFINE_TREE.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

if(nargin ~= 2)
  error('Requires exactly 2 arguments.');
end

% Allow direct input of an htensor (only field children is used)
if(isa(children, 'htensor'))
  children = children.children;
elseif( ~isnumeric(children) || size(children, 2) ~= 2 || ...
       any(any(~ismember(children, 0:size(children, 1)))) )
  error(['First argument CHILDREN must be an N x 2 array of integers' ...
	 ' between 0 and N, defining a binary tree.']);
end

% Check input ii
if(~(isindexvector(ii) && isscalar(ii)) || ii > size(children, 1))
  error(['Node index ii must be a positive non-zero integer, and' ...
	 ' cannot exceed the number of nodes.']);
end

sub_ind = ii;
ind = 1;

while(length(sub_ind) >= ind && length(sub_ind) <= size(children, 1))
  
  if(all(children(sub_ind(ind), :) == [0 0]))
    %do nothing
  else
    sub_ind = [sub_ind(1:ind), children(sub_ind(ind), :), sub_ind(ind+1:end)];
  end
  ind = ind + 1;
  
end

if(length(sub_ind) ~= length(unique(sub_ind)))
  error('The tree defined by CHILDREN contains a loop.');
end
