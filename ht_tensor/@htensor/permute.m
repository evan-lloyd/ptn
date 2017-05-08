function x = permute(x, order)
%PERMUTE Permute dimensions of an htensor.
%
%   B = PERMUTE(A, ORDER) rearranges the dimensions of A so that they
%   are in the order specified by the vector ORDER. The htensor
%   produced has the same values of A but the order of the subscripts
%   needed to access any particular element is rearranged as
%   specified by ORDER.
%
%   Note that the dimension tree of A is not modified; only which
%   dimensions are represented by each node changes.
%
%   See also HTENSOR/IPERMUTE.
%

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
  error('Second argument ORDER must be a vector of indexes.');
end

d = ndims(x);

% Check permutation given by order
if( ~isequal(sort(order, 'ascend'), 1:d) )
  error(['Invalid permutation, ORDER must be a permutation of 1:' ...
	 ' d.']);
end

% Calculate the inverse permutation
inv_order(order) = 1:d;

% Apply inverse permutation to dims at each node
for ii=1:x.nr_nodes
  x.dims{ii} = inv_order(x.dims{ii});
end
