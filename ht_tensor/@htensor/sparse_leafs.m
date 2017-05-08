function x = sparse_leafs(x)
%SPARSE_LEAFS Converts leaf matrices U to sparse matrices.
%
%  Y = SPARSE_LEAFS(X) results in an htensor identical to X, with
%  sparse leaf matrices U{i}.
%
%  See also: FULL_LEAFS.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

for ii=1:x.nr_nodes
  if(x.is_leaf(ii))
    x.U{ii} = sparse(x.U{ii});
  end
end