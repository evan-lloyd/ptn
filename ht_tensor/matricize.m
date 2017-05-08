function A = matricize(x, row_dims, col_dims)
%MATRICIZE Returns the matricization of a (full) tensor.
%
%   A = matricize(X, ROW_DIMS, COL_DIMS) returns the matricization
%   of X into a size(ROW_DIMS) x size(COL_DIMS) matrix. Together,
%   ROW_DIMS and COL_DIMS must contain all 1:ndims(X)
%   dimensions. They may contain an arbitrary number of trailing
%   singleton dimensions, but these have no influence on the result.
%
%   A = matricize(X, ROW_DIMS) calculates COL_DIMS such that the
%   above condition is satisfied. The entries of COL_DIMS are
%   arranged in ascending order.
%
%   Note that matricize(X, 1:ndims(X)) corresponds to X(:).
%
%   See also DEMATRICIZE.
%

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

% Generate col_dims if not initialized: all remaining dimensions in
% ascending order

% Check first two input arguments
if(nargin <= 1)
  error('Insufficient number of arguments; need at least 2.');
elseif(~isnumeric(x))
  error('First argument must be numeric.');
elseif(~isindexvector(row_dims) || ...
       numel(unique(row_dims)) ~= numel(row_dims))
  error('ROW_DIMS must be an index vector without double entries.');
else
  % Remove trailing singleton dimensions
  row_dims = row_dims(row_dims <= ndims(x));
end

% Check third (optional) argument
if(nargin == 2)
  col_dims = setdiff(1:ndims(x), row_dims);
elseif(isindexvector(col_dims) && numel(unique(col_dims)) == numel(col_dims))
  % Remove trailing singleton dimensions
  col_dims = col_dims(col_dims <= ndims(x));
else
  error('COL_DIMS must be an index vector without double entries.')
end

% Check consistency of input dimensions
if( ~isequal( sort([row_dims, col_dims]), 1:ndims(x) ) )
  error('Dimensions ROW_DIMS, COL_DIMS do not match tensor x.')
end

% Save size of original tensor
sz = size(x);

% Permute dimensions of tensor x
x = permute(x, [row_dims, col_dims]);

% Reshape to matrix
A = reshape(x, prod(sz(row_dims)), prod(sz(col_dims)) );
