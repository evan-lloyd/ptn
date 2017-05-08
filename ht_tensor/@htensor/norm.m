function nrm = norm(x)
%NORM Tensor norm of an htensor.
%
%   NORM(X) returns the tensor norm of an htensor.
%
%   If X is orthogonal, this is calculated as ||B{root_node}||,
%   otherwise corresponds to abs(sqrt(innerprod(X, X))).
%
%   See also HTENSOR, INNERPROD.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

% For orthogonal x, only B{1} (at the root node) is needed.
if(x.is_orthog)
  nrm = norm(x.B{1}, 'fro');
  return;
end

M = cell(x.nr_nodes, 1);

% Start at leaves, move up the levels
for ii=x.nr_nodes:-1:1
  
  if(x.is_leaf(ii))
    % M_t = U_t' * U_t
    M{ii} = x.U{ii}'*x.U{ii};
  else
    ii_left  = x.children(ii, 1);
    ii_right = x.children(ii, 2);
    
    % M_t = B_t' * (M_t1 kron M_t2) * B_t
    % (interpreting B_t to be in matricized form)
    B_ = ttm(x.B{ii}, { M{ii_left}, M{ii_right} }, [1 2]);
    M{ii} = ttt(x.B{ii}, B_, [1 2]);
    
    % Save memory
    M{ii_left} = []; M{ii_right} = [];
  end
end

nrm = abs(sqrt(M{1}));
