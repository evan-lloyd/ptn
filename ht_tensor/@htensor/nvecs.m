function U = nvecs(x, dim, r)
%NVECS Computes leading mode-n vectors of htensor.
%
%   U = NVECS(X, DIM, R) returns the R leading mode-DIM vectors of
%   htensor X. They correspond to the leading left singular vectors
%   of the matricization X^(DIM).
%
%   See also HTENSOR, ORTHOG, GRAMIANS
%

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

if(nargin ~= 3)
  error('Requires exactly 3 arguments.');
end

if(~isa(x, 'htensor'))
  error('First argument must be of class htensor.');
end

if(~(isindexvector(dim) && isscalar(dim)) || dim > ndims(x))
  error(['Second argument DIM must be an index, cannot be bigger' ...
	' than ndims(x).']);
end

if(~(isindexvector(dim) && isscalar(dim)))
  error('Third argument R must be an integer.');
end

x = orthog(x);

% Calculate the gramians of orthogonalized x
G = gramians(x);

% Leaf node for dimension dim
ii = x.dim2ind(dim);

% Calculate left singular values at node ii
[U, ~] = htensor.left_svd_gramian(G{ii});

U(:, end+1:r) = 0;

U = U(:, 1:r);

U = x.U{ii}*U;