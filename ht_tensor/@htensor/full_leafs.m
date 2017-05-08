function x = full_leafs(x)
%FULL_LEAFS Converts leaf matrices U to dense matrices.
%
%  Y = FULL_LEAFS(X) results in an htensor identical to X, with
%  dense leaf matrices U{i}.
%
%  See also: SPARSE_LEAFS.
% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

for ii=1:x.nr_nodes
  if(x.is_leaf(ii))
    x.U{ii} = full(x.U{ii});
  end
end