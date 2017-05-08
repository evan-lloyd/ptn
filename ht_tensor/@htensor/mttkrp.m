function M = mttkrp(x, U, n)
%MTTKRP Matricized tensor times Khatri-Rao product for htensor.
%
%   V = MTTKRP(X,U,n) calculates the matrix product of the n-mode
%   matricization of X with the Khatri-Rao product of all entries in
%   U, a cell array of matrices, except the nth.
%
%   Uses transposed matrices U, not complex conjugate transposed U.
%
%   See also HTENSOR, HTENSOR/TTM

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

if(nargin ~= 3)
  error('Requires exactly 3 arguments.');
end

if(~isa(x, 'htensor'))
  error('First argument must be of class htensor.');
end

if(~isa(U, 'cell') || ~all(cellfun( ...
    @(x)((isnumeric(x) && ndims(x) == 2) || isa(x, 'function_handle')), U)))
  error(['Second argument U must be a cell array of matrices or' ...
	 ' function handles.']);
end

if(~(isindexvector(n) && isscalar(n)))
  error('Third argument N must be a positive integer.')
end

% Order and dimensions of x
d = ndims(x);
sz = size(x);

% Check order of input
if(ndims(x) ~= length(U))
  error(['Number of elements in cell array U must be equal to the' ...
	 ' order of tensor X.']);
end

m1 = cellfun('size', U, 1);
m2 = cellfun('size', U, 2);

% Check number of rows of U
if( any(sz([1:n-1,n+1:d]) ~= m1([1:n-1,n+1:d])) )
  error('Dimension mismatch.');
end

% Check number of columns of U
if( any(diff(m2([1:n-1,n+1:d])) ~= 0) )
  error(['All matrices in cell array U must have the same number' ...
	 ' of columns.'])
end

% Number of columns of the matrices U
m = m2(1);

% Apply U{ii}' to x in all dimensions except n
x = ttm(x, U, -n, 't');

% Go through the columns of matrix M to evaluate the Khatri-Rao
% product (equivalent to Kronecker product in each column vector)
M = zeros(sz(n), m);
for jj=1:m
  ind = jj*ones(d, 2);
  ind(n, :) = [1, sz(n)];
  M(:, jj) = full_block(x, ind);
end
