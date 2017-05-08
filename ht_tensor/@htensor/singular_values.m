function sv = singular_values(x)
%SINGULAR_VALUES Singular values of matricization at each node of htensor.
%
%   S = SINGULAR_VALUES(X) calculates the singular value tree of x,
%   returning the cell array S containing the singular values at each
%   node except the root node.
%
%   See also HTENSOR, ORTHOG, GRAMIANS
%

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

% Calculate the gramians of orthogonalized x
G = gramians(orthog(x));

% Go through dimension tree
for ii=2:x.nr_nodes
  
  % calculate the left singular vectors U_ and singular values s
  % of X_{ii} from the gramian G{ii}.
  [tmp, sv{ii}] = htensor.left_svd_gramian(G{ii});
end