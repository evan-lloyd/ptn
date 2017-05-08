function x = dematricize(A, sz, row_dims, col_dims)
%DEMATRICIZE Finds the original (full) tensor from its matricization.
%
%   X = dematricize(A, SZ, ROW_DIMS, COL_DIMS) returns the tensor X
%   of size SZ. This is the reverse of 
%     A = matricize(X, ROW_DIMS, COL_DIMS)
%   Together, ROW_DIMS and COL_DIMS must contain all 1:d
%   dimensions, and the matrix A must be of size SZ(ROW_DIMS) x
%   SZ(COL_DIMS). ROW_DIMS and COL_DIMS may contain an arbitrary
%   number of trailing singleton dimensions, but these have no
%   influence on the result.
%
%   X = dematricize(A, SZ, ROW_DIMS) calculates COL_DIMS such that the
%   above condition is satisfied. The entries of COL_DIMS are
%   arranged in ascending order.
%
%   Examples:
%   x = rand(3, 4, 5);
%   A = matricize(x, [3 2], 1)
%   size(A) % [20, 3]
%   x_ = dematricize(A, [3 4 5], [3 2], 1);
%   
%   See also DEMATRICIZE.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

% Generate col_dims if not initialized: all remaining dimensions in
% ascending order

% Check first two input arguments
if(nargin <= 2)
  error('Insufficient number of arguments; need at least 3.');
elseif(~isnumeric(A) )
  error('First argument must be numeric.');
elseif( ~isvector(sz) || any(floor(sz) ~= ceil(sz)) || any(sz < 0) )
  error(['Second argument must be a vector of real non-negative' ...
	 ' integers.']);
elseif(~isindexvector(row_dims) || numel(unique(row_dims)) ~= numel(row_dims))
  error('ROW_DIMS must be an index vector without double entries.');
end

% Insert singleton dimensions if numel(sz) < 2
sz(end+1:2) = 1;

% Remove trailing singleton dimensions
d = numel(sz);
row_dims = row_dims(row_dims <= d);

% Check fourth (optional) argument
if(nargin == 3)
  col_dims = setdiff(1:d, row_dims);
elseif(isindexvector(col_dims) && numel(unique(col_dims)) == numel(col_dims))
  % Remove trailing singleton dimensions
  col_dims = col_dims(col_dims <= d);
else
  error('COL_DIMS must be an index vector without double entries.')
end

% Check consistency of input dimensions
sorted_dims = sort([row_dims, col_dims]);
if( ~isequal( sorted_dims, 1:d ) )
  error('Dimensions ROW_DIMS, COL_DIMS do not match size SZ.');
end

% Reshape to tensor
x = reshape(A, sz([row_dims, col_dims]) );

% Inverse permute
x = ipermute(x, [row_dims, col_dims]);
