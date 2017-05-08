function G = gramians_nonorthog(x)
%GRAMIANS_NONORTHOG Gramians of matricization at each node of an htensor.
%
%   G = GRAMIANS_NONORTHOG(X) returns a cell-array containing the reduced
%   gramians G_t = Y_t*Y_t'.
%
%   In contrast to GRAMIANS, x is not orthogonalized at the start
%   of this method.
%
%   Node t's matricization of X can be written as
%     X_t = U_t * Y_t,
%   where U_t is assembled from the descendants of node t. The
%   Gramian of the t-unfolding of X is
%   X_t * X_t' = U_t * G_t * U_t'.
%
%   See also HTENSOR

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

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
    % (interpreting B_t to be in matricized form)
    B_ = ttm(x.B{ii}, { M{ii_left}, M{ii_right} }, [1 2]);
    M{ii} = ttt(x.B{ii}, B_, [1 2]);
    
  end
end

G = cell(1, x.nr_nodes);
G{1} = 1;

% Start from root node, move down.
for ii=find(x.is_leaf == false)
  
  % Child nodes
  ii_left  = x.children(ii, 1);
  ii_right = x.children(ii, 2);
  
  % Calculate < B{ii}, B{ii} x_1 G{ii} >_(1, 2) and _(1, 3)
  B_mod = ttm(conj(x.B{ii}), G{ii}, 3);
  
  B_mod_left  = ttm(B_mod, M{ii_right}, 2);
  B_mod_right = ttm(B_mod, M{ii_left} , 1);
  
  G{ii_left } = ttt(conj(x.B{ii}), B_mod_left, [2 3]);
  G{ii_right} = ttt(conj(x.B{ii}), B_mod_right, [1 3]);

end
