function G = gramians(x)
%GRAMIANS Gramians of matricization at each node of an htensor.
%
%   G = GRAMIANS(X) returns a cell-array containing the reduced
%   gramians G_t = Y_t'*Y_t.
%
%   The tensor X is required to be orthogonal. Then, the node t's
%   matricization of X can be written as
%     X_t = U_t * Y_t',
%   where U_t is assembled from the descendants of node t. As X is
%   orthogonal, the matrix U_t must be column-orthogonal, and the
%   Gramian of the t-unfolding of X is
%   X_t * X_t' = U_t * G_t * U_t'.
%
%   This method assumes x to be orthogonal. If X is a
%   non-orthogonal htensor, it is orthogonalized and a warning
%   message is output.
%
%   See also HTENSOR

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

% Check if x is orthogonal
if(~x.is_orthog)
  x = orthog(x);
  disp('htensor.gramians: Warning, non-orthogonal argument.')
end

G{1} = 1;

% Start from root node, move down.
for ii=find(x.is_leaf == false)

  % Child nodes
  ii_left  = x.children(ii, 1);
  ii_right = x.children(ii, 2);
  
  % Calculate < B{ii}, B{ii} x_1 G{ii} >_(1, 2) and _(1, 3)
  B_mod = ttm(conj(x.B{ii}), G{ii}, 3);
  
  G{ii_left } = ttt(conj(x.B{ii}), B_mod, [2 3]);
  G{ii_right} = ttt(conj(x.B{ii}), B_mod, [1 3]);

end
