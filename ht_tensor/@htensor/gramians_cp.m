function G = gramians_cp(cp, weights, tree_type)
%GRAMIANS_CP Gramians of matricization at each node of a CP tensor.
%
%   G = GRAMIANS_CP(CP) returns a cell-array containing the reduced
%   gramians G_t = Y_t*Y_t' of the tensor defined by
%     SUM_i W(i) CP{d}(:, i) x ... CP{1}(:, i).
%
%   Node t's matricization of X can be written as
%     X_t = U_t * Y_t,
%   where U_t is assembled from the descendants of node t. The
%   Gramian of the t-unfolding of X is
%   X_t * X_t' = U_t * G_t * U_t'.
%
%   See also GRAMIANS, GRAMIANS_NONORTHOG, HTENSOR

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

if(nargin == 1)
  if(isa(cp, 'ktensor'))
    x = htensor(cp);
    weights = cp.lambda;
  elseif(isa(cp, 'cell'))
    weights = ones(size(cp{1}, 2), 1);
    x = htensor(cp, weights);
  end
elseif(nargin == 2)
  x = htensor(cp, weights);
elseif(nargin == 3)
  x = htensor(cp, weights, tree_type);
else
  error('Requires 1 or 2 arguments.');
end

if(size(weights, 2) ~= 1)
  weights = transpose(weights);
end

% Child nodes
root_left  = x.children(1, 1);
root_right = x.children(1, 2);

% Calculate M{ii} = x.U{ii}'*x.U{ii}:
M = cell(1, x.nr_nodes);

% Start at leaves, move up the levels
for ii=x.nr_nodes:-1:2
  
  if(x.is_leaf(ii))
    % M_t = U_t' * U_t
    M{ii} = full(x.U{ii}'*x.U{ii});
  else
    ii_left  = x.children(ii, 1);
    ii_right = x.children(ii, 2);
    
    % M_t = B_t' * (M_t1 kron M_t2) * B_t
    M{ii} = M{ii_left}.*M{ii_right};
    
  end
end

G = cell(1, x.nr_nodes);
G{1} = 1;

% Calculate < B{ii}, B{ii} x_1 G{ii} >_(1, 2) and _(1, 3)
G{root_left}  = (weights*weights').*M{root_right};
G{root_right}  = (weights*weights').*M{root_left};

% Start from root node, move down.
for ii=find(x.is_leaf == false)
  
  if(ii == 1)
    continue;
  end
  
  % Child nodes
  ii_left  = x.children(ii, 1);
  ii_right = x.children(ii, 2);
  
  % Calculate < B{ii}, B{ii} x_1 G{ii} >_(1, 2) and _(1, 3)
  G{ii_left } = G{ii}.*M{ii_right};
  G{ii_right} = G{ii}.*M{ii_left};

end
