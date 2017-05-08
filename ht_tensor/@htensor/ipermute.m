function x = ipermute(x, order)
%IPERMUTE Inverse permute dimensions of an htensor.
%
%   A = IPERMUTE(B, ORDER) is the inverse of PERMUTE. The dimensions
%   of A are rearranged such that B = PERMUTE(A, ORDER). The htensor
%   produced has the same values as A but the order of the subscripts
%   needed to access any particular element is rearranged.
%
%   Note that the dimension tree of A is not modified; only which
%   dimensions are represented by each node changes (i.e. the field
%   X.DIMS).
%
%   If xp == permute(x, order), then x == ipermute(xp, order).
%
%   See also HTENSOR/PERMUTE.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

if(~isa(x, 'htensor'))
  error('First argument must be of class htensor.');
end

if(~isindexvector(order))
  error('Second argument ORDER must be an index vector.');
end

% Check permutation given by order
if( ~isequal(sort(order, 'ascend'), 1:ndims(x)) )
  error(['Invalid permutation, ORDER must be a permutation of 1:' ...
	 ' d.']);
end

inverseorder(order) = 1:numel(order);   % Inverse permutation order

x = permute(x, inverseorder);
