function r = rank(x, ii)
%RANK Hierarchical ranks of an htensor.
%
%   Returns a (2*NDIMS(X)-1)-vector containing the hierarchical ranks
%   of htensor X.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

rk_leafs = cellfun('size', x.U, 2);
rk_nodes = cellfun('size', x.B, 3);
rk_nodes(x.is_leaf) = 0;

r = rk_leafs + rk_nodes;

if nargin > 1
  r = r(ii);
end
