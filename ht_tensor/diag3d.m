function x = diag3d(v)
%DIAG3D Returns a third order diagonal tensor
%
%   X = DIAG3D(V) returns the 3rd-order MDA X with X(i, i, i) = v(i).
%   
% Example:
% >> x = diag3d([2 3])
%  x(:,:,1) =
%       2     0
%       0     0
%  x(:,:,2) =
%       0     0
%       0     3

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

% Check v
if(~isnumeric(v) || ~isvector(v))
  error('Argument v must be a (numeric) vector.');
end

% Construct n x n x n - array x
n = length(v);
x = zeros([n n n]);

% Fill in diagonal entries
for ii=1:n
  x(ii, ii, ii) = v(ii);
end