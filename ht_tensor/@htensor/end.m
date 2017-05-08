function e = end(x, k, n)
%END Last index in one dimension of an htensor.
%
%   The expression X(end,:,:) calls END(X,1,3) to determine
%   the value of the first index.
%
%   See also HTENSOR, HTENSOR/SUBSREF, END.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

% Check that n == ndims(x)
if( n ~= ndims(x) )
  error('Wrong number of indexes: ndims(X) = %d.', ndims(x));
end

% Check that k is an integer between 1 and n
if( floor(k) ~= ceil(k) || k > n || k < 1 )
  error('Subscript indexes K must be real positive integers.')
end

e = size(x, k);
