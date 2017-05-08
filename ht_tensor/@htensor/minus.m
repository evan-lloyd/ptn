function x = minus(x1, x2)
%MINUS Binary subtraction for htensor.
%
%   X = MINUS(X1,X2) subtracts an htensor from another of the same size.
%   The result is an htensor of the same size.
%
%   See also HTENSOR.
%

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

% Check type of x1, x2
if( ~isa(x1, 'htensor') || ~isa(x2, 'htensor') )
  error('X1 and X2 must be of class htensor.');
end

% Check compatibility of dimension trees.
if(~equal_dimtree(x1, x2))
  error('Dimension trees of X1 and X2 differ.');
end

% Check sizes
if(~isequal(size(x1), size(x2)))
  error('Tensor dimensions must agree.')
end

x = plus(x1, -x2);