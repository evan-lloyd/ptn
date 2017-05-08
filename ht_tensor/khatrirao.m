function Z = khatrirao(X, Y)
%KHATRIRAO Khatri-Rao product.
%
%   Z = KHATRIRAO_T(X, Y) Computes the transposed Khatri-Rao
%   product, Z = X (k-r) Y.
%
%   The input matrices X and Y must be an NxM- and NxK-matrix,
%   making the result Z an Nx(M*K)-matrix with
%
%      Z(:, j + p*(i-1)) = X(:, i).*Y(:, j)
%   
%   or, equivalently
%
%      Z(i, :) = kron(X(i, :), Y(i, :))
%
%   See also: KHATRIRAO, HTENSOR, ELEM_MULT.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

[m, n] = size(X);
[p, n_] = size(Y);

if(n ~= n_)
  error('Wrong dimensions.');
end

Z = zeros(m*p, n);

for ii=1:n
  C = Y(:, ii)*X(:, ii).';
  Z(:, ii) = C(:);
end

% slower (tested with 40x200, 30x200 and 200x40, 150x40)
%for ii=1:m
%  for jj=1:p
%    Z(jj + p*(ii-1), :) = X(ii, :).*Y(jj, :);
%  end
%end

