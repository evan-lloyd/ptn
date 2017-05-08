function x = change_root(x, ind, lr_subtree)
%CHANGE_ROOT Changes the root of the dimension tree.
%
%   Y = CHANGE_ROOT(X, IND) changes the root of a dimension tree to
%   the node indicated by IND. This is possible for every index. The
%   entries of X and Y are identical, only the dimension tree
%   changes. Node IND becomes the right child of the root in Y (and
%   its index changes from IND to 3).
%
%   Y = CHANGE_ROOT(X, IND, LR_SUBTREE) does the same, but leaves
%   the choice of the subtree's side to the user. With option
%   'left', the node IND becomes the left child of the root node,
%   otherwise the right child.
%
%   Special case for IND = 1 (root node): Instead of leaving the
%   tree as it is, an additional level (and a singleton dimension)
%   is added:
%                        1
%                      /   \
%                    1    original dimension tree
%
%   Example:
%   x = htensor([4 6 2 3]);
%   y = change_root(x, 6);
%   size(y)
%     [4 6 2 3]
%   z = change_root(x, 1);
%   size(z)
%     [1 4 6 2 3]
%
%   dimension trees:
%
%   x:           y:   / \     z:   /  \
%       / \         / \  3        1   / \
%     /\   /\      /\  4            /\   /\ 
%    1  2 3  4    1  2             2  3 4  5
%
%   See also HTENSOR.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

if(~isa(x, 'htensor'))
  error('First argument must be of class htensor.');
end

if(nargin == 1)
  error('Requires at least 2 arguments.')
end

if(~isindexvector(ind) || ~isscalar(ind))
  error('Second argument IND must be an integer (node index).')
end

if(nargin == 2)
  lr_subtree = 'right';
elseif(~ischar(lr_subtree))
  error('LR_SUBTREE must be a char array.')
end

if(ind ~= 1)
  
  children = x.children;
  
  p = x.parent;
  nr_nodes = size(children, 1);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % apply B{1} to child node, put  identity matrix in B{1} %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  if(x.rank(x.children(1, 1)) ~= x.rank(x.children(1, 2)))
    [k, left_right] = min(x.rank(x.children(1, :)));
  else
    left_right = 1;
    k = x.rank(x.children(1, 1));
  end
  
  B_mat = matricize(x.B{1}, left_right);
  
  ii_child = x.children(1, 3-left_right);
  if(~x.is_leaf(ii_child))
    x.B{ii_child} = ttm(x.B{ii_child}, B_mat, 3);
  else
    x.U{ii_child} = x.U{ii_child}*B_mat.';
  end
  
  x.B{1} = eye(k);

  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Insert additional node into the tree   %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  new_node = nr_nodes + 1;
  
  % is ind right or left child?
  left_right = find(children(p(ind), :) == ind);
  
  % ind's parent has child nr_nodes+1
  children(p(ind), left_right) = new_node;
  p(new_node) = p(ind);
  
  % new node has children ind
  children(new_node, :) = [ind, ind];
  p(ind) = new_node;
  
  % insert identity tensor at new node
  x.B{new_node} = eye(x.rank(ind));
  x.U{new_node} = [];
  x.dims{new_node} = [];
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Change parent-child direction between ancestors of ind %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ii = new_node;
  ii_par = p(ii);
  if(strcmp(lr_subtree, 'left'))
    children(ii, 2) = ii_par;
  else
    children(ii, 1) = ii_par;
  end
  
  while(ii_par ~= 1)
    
    left_right = find(children(ii_par, :) == ii);
    ii = ii_par;
    ii_par = p(ii);
    
    children(ii, left_right) = ii_par;
    
    if(left_right == 1)
      x.B{ii} = permute(x.B{ii}, [3 2 1]);
    else
      x.B{ii} = permute(x.B{ii}, [1 3 2]);
    end
    
  end
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Eliminate old root node from tree     %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  left_right = find(children(1, :) == ii);
  % right_left: 1 if left_right == 2, 2 if left_right == 1
  right_left = 3 - left_right;
  ii_brother = children(1, right_left);
  
  children(1, left_right) = ii_brother;
  children(1, :) = [0 0];
  
  subs = find(children == 1);
  children(subs) = ii_brother;
  
  %B_mat = matricize(x.B{1}, left_right);
  %if(~x.is_leaf(ii_brother))
  %  x.B{ii_brother} = ttm(x.B{ii_brother}, B_mat, 3);
  %else
  %  x.U{ii_brother} = x.U{ii_brother}*B_mat.';
  %end
  
  x.B{1} = [];
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Change indexes, make new array children %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ii = 1;
  nodes_list = new_node;
  
  while(length(nodes_list) < nr_nodes)
    
    if( all(children(nodes_list(ii), :) ~= [0 0]) )
      if(diff(children(nodes_list(ii), :)) ~= 0)
	nodes_list = [nodes_list, children(nodes_list(ii), :)];
      else
	nodes_list = [nodes_list, children(nodes_list(ii), 1)];
      end
    end
    
    ii = ii+1;
  end
  
  old2new = nodes_list;
  new2old(old2new) = 1:nr_nodes;
  
  children = children(old2new, :);
  for ii=1:size(children, 1)
    if( any(children(ii, :) ~= 0) )
      children(ii, :) = new2old(children(ii, :));
    end
  end
  
  x.children = children;
  
  x.dims = x.dims(old2new);
  
  x.U = x.U(old2new);
  x.B = x.B(old2new);
  x.is_orthog = false;
  
  % Adjust dims to new tree structure
  for ii=x.nr_nodes:-1:1
    if(~x.is_leaf(ii))
      ii_left  = x.children(ii, 1);
      ii_right = x.children(ii, 2);
      x.dims{ii} = [x.dims{ii_left}, x.dims{ii_right}];
    end
  end
  
  
else % if ind == 1 (root node)                1
     % created additional level above root:  /  \
     %                                      1  orig_tree
     
  children = x.children;
  children(children ~= 0) = children(children ~= 0) + 2;
  
  if(strcmp(lr_subtree, 'left'))
    children = [3 2; 0 0; children];
  else  
    children = [2 3; 0 0; children];
  end
  
  dims((1:x.nr_nodes)+2) = x.dims;
  dims = cellfun(@(var)(var+1), dims, 'UniformOutput', false);
  
  if(strcmp(lr_subtree, 'left'))
    dims{1} = [dims{3}, 1];
  else
    dims{1} = [1, dims{3}];
  end
  dims{2} = 1;
  
  U((1:x.nr_nodes)+2) = x.U;
  U{2} = 1;
  
  B((1:x.nr_nodes)+2) = x.B;
  B{1} = 1;
  
  x = htensor(children, dims, U, B, false);
  
end
