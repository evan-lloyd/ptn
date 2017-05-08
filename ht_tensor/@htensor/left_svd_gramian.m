function [U, s] = left_svd_gramian(G)
%LEFT_SVD_GRAMIAN Left singular vectors and singular values from Gramian.
%
%   LEFT_SVD_QR(G) Efficiently computes the left singular vectors
%   and the singular values of matrix A, where G = A*A'.
%

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

if(~isnumeric(G) || ndims(G) ~= 2 || size(G, 1) ~= size(G, 2))
  error('G must be a quadratic matrix.');
end

% Symmetrize matrix G
G = (G + G')/2;

% Calculate eigenvalue decomposition
[U, S] = svd(G);

% Calculate singular values from eigenvalues of the gramian
s = sqrt(abs(diag(S)));

% sort singular values and vectors
[s, idx] = sort(s, 'descend');
U = U(:, idx);
