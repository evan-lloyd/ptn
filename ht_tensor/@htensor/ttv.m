function x = ttv(x, v, dims)
%TTV Tensor-times-vector for htensor.
%
%   Y = TTV(X,A,N) computes the product of htensor X with a (column)
%   vector A, along mode N. The integer N specifies the dimension in X
%   along which A is multiplied. The arguments must satisfy size(A, 1)
%   = size(X,N). Note that ndims(Y) = ndims(X) - 1 because the N-th
%   dimension is removed. This is in contrast to TTM(X, A', DIMS),
%   which leaves a singleton dimension.
%
%   Y = TTV(X,{A,B,C,...}) computes the product of htensor X with the
%   sequence of vectors in the cell array. The products are computed
%   sequentially along all dimensions (or modes) of X. The cell array
%   must contain ndims(X) vectors.
%
%   Y = TTV(X,{A,B,C,...},DIMS) computes the sequence of htensor-vector
%   products along the dimensions specified by DIMS.
%
%   In the complex case, the vectors A, B, ... are not put to the
%   complex conjugate.
%
%   Examples
%   X = htenrandn([5,3,4,2]);
%   a = rand(5,1); b = rand(3,1); c = rand(4,1); d = rand(2,1);
%   Y = ttv(X, a, 1) %<-- X times a in mode 1
%   Y = ttv(X, {a,b,c,d}, 1) %<-- same as above
%   Y = ttv(X, {a,b,c,d}, [1 2 3 4]) %<-- All-mode multiply
%   Y = ttv(X, {d,c,b,a}, [4 3 2 1]) %<-- same as above
%   Y = ttv(X, {a,b,c,d}) %<-- same as above
%   Y = ttv(X, {c,d}, [3 4]) %<-- X times c in mode-3 & d in mode-4
%   Y = ttv(X, {a,b,c,d}, [3 4]) %<-- same as above
%   Y = ttv(X, {a,b,d}, [1 2 4]) %<-- 3-way multiplication
%   Y = ttv(X, {a,b,c,d}, [1 2 4]) %<-- same as above
%   Y = ttv(X, {a,b,d}, -3) %<-- same as above
%   Y = ttv(X, {a,b,c,d}, -3) %<-- same as above
%
%   See also HTENSOR, HTENSOR/TTT, HTENSOR/TTM, HTENSOR/SQUEEZE.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

% Check number of arguments
if(nargin < 2)
  error('Requires at least 2 arguments.');
end

% Check class of x
if(~isa(x, 'htensor'))
  error('First argument X must be of class htensor.');
end

% Default values of optional arguments
if(nargin == 2)
  dims = 1:ndims(x);
end

% Case of only one vector: Put it into a cell array.
if (isa(v, 'cell'))
  % everything ok
elseif(isvector(v))
  v = {v};
else
  error('V must be a vector or a cell array of vectors.')
end

% Check dims:
if(~isindexvector(dims) && ~isindexvector(-dims))
  error('DIMS must be a vector of indexes.');
end

% If dims is negative, select all dimensions except those indicated
if( all(dims < 0) )
  dims = setdiff(1:ndims(x), -dims);
end

% Check that dims are valid
if( ~all(ismember(dims, 1:ndims(x))) || numel(dims) ~= numel(unique(dims)) )
  error('Invalid argument DIMS.');
end

%  Loop over cell array v
for ii=1:length(dims)
  ind = x.dim2ind(dims(ii));
  
  if(size(v{ii}, 1) ~= size(x, dims(ii)))
    error('Matrix dimensions must agree.')
  end
  x.U{ind} = v{ii}.'*x.U{ind};
end

x.is_orthog = false;

x = squeeze(x, dims);
