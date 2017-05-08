function d = ndims(x)
%NDIMS Order (number of dimensions) of an htensor.
%
%   NDIMS(X) returns the order of htensor X.
%
%   Examples
%   X = htenrandn([4,3]); ndims(X) %<-- Returns 2
%   X = htenrandn([4 3 1]); ndims(X) %<-- Returns 3
%
%   See also HTENSOR

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

d = numel(x.dims{1});