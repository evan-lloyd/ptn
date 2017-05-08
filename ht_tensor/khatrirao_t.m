function Z = khatrirao_t(X, Y)
%KHATRIRAO_T Transposed Khatri-Rao product.
%
%   Z = KHATRIRAO_T(X, Y) Computes the transposed Khatri-Rao
%   product, Z.' = X.' (k-r) Y.'.
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
%   See also: khatrirao, HTENSOR, ELEM_MULT.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

[n, m] = size(X);
[n_, p] = size(Y);

if(n ~= n_)
  error('Wrong dimensions.');
end

Z = zeros(n ,m*p);

for ii=1:m
  for jj=1:p
    Z(:, jj + p*(ii-1)) = X(:, ii).*Y(:, jj);
  end
end

% slower (tested with 200x40, 200x30 and 40x200, 40x150)
%for ii=1:n
%  C = Y(ii, :).'*X(ii, :);
%  Z(ii, :) = C(:);
%end
