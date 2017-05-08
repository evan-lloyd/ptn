function is_iv = isindexvector(ind)
%ISINDEXVECTOR True if input is a vector of indexes.
%
%   ISINDEXVECTOR(IND) is true if IND is a vector of integers that
%   are greater than or equal to zero.
%
%   Utility function for argument checking.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

is_iv = isnumeric(ind) && isvector(ind) && ...
	all(floor(ind) == ceil(ind)) && ...
	all(ind > 0);

if(isnumeric(ind) && isempty(ind))
  is_iv = true;
end
  