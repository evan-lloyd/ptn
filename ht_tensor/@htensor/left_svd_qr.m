function [U, s] = left_svd_qr(A)
%LEFT_SVD_QR Left singular vectors and singular values.
%
%   LEFT_SVD_QR(A) Efficiently computes the left singular vectors
%   and the singular values of matrix A.
%

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

if(~isnumeric(A) || ndims(A) ~= 2)
  error('First argument A must be a matrix.');
end

if(size(A, 1) > size(A, 2))

  [Q, R] = qr(A, 0);
  [U, S, ~] = svd(R, 0);
  U = Q*U;
  
else
  
  R = qr(A');
  R = triu(R);
  R = R(1:size(R, 2), :);
  %size(R')
  [U, S, ~] = svd(R', 0);
  
end

s = diag(S);
