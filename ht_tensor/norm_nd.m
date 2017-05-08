function nrm = norm_nd(x)
%NORM_ND Tensor norm of a multidimensional array.
%
%   NORM(X) returns the tensor norm of a multidimensional
%   array. For a matrix, this is the tensor norm.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

% Check X
if(~isnumeric(x))
  error('X must be a MATLAB (multidimensional) array.');
end

% Calculate tensor norm
nrm = norm(x(:));