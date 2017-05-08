function y = power(x, p)
%POWER Elementwise square of tensor X.
%
%   Y = TIMES(X, P) For P == 2, returns the elementwise square of X
%   (i.e., the elementwise product of X with itself. For all other
%   P, returns an error message.
%
%   See also HTENSOR/TIMES.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

if(nargin ~= 2)
  error('Exactly two arguments required');
end

if(~isa(x, 'htensor'))
  error('First argument must be an htensor');
end

if(p ~= 2)
  error('Only power(x, 2) (i.e x.^2) is supported');
end

y = x.*x;