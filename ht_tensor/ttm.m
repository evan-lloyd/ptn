function x = ttm(x, A, varargin)
%TTM Tensor times matrix (full tensor).
%
%   Y = TTM(X, A, N) computes the tensor-times-matrix product in
%   mode N,
%      Y = X x_N A.
%   The arguments must satisfy size(X, N) == size(A, 2), and the
%   result Y will have size(Y, N) == size(A, 1).
%
%   Y = TTM(X, A) where A is a cell array with D >= ndims(X)
%   entries, computes
%     Y = X x_1 A{1} x_2 A{2} ... x_D A{D}
%   If any entries of A are empty, nothing is done in that
%   dimension.
%
%   Y = TTM(X, A, EXCLUDE_DIMS) where A is a cell array with D >=
%   NDIMS entries, and EXCLUDE_DIMS contains negative integer(s),
%   computes 
%     Y = X
%     Y = Y x_i A{i} for i from 1 to D, i not in -EXCLUDE_DIMS.
%
%   Y = TTM(X, A, DIMS) where A is a cell array and DIMS contains
%   positive integers, with numel(A) == numel(DIMS). Computes
%     Y = X x_DIMS(1) A{1} x_DIMS(2) A{2} ... x_DIMS(end) A{end}
%   Note that each dimension can only appear once in DIMS.
%
%   Y = TTM(...,'t') performs the same computations as above, but
%   uses transposed matrices.
%
%   Y = TTM(...,'h') performs the same computations as above, but
%   uses conjugate transposed matrices.
%
%   Each matrix in A can be replaced by a function handle that
%   applies a linear function to its input vectors.
%
%   Note that the usage is slightly different from the function ttm in
%   the Tensor Toolbox, as MATLAB multidimensional arrays do not have
%   fixed order.
%
%   Examples
%   X = rand([5,3,4,2]);
%   A = rand(4,5); B = rand(4,3); C = rand(3,4); D = rand(3,2); 
%   Y = ttm(X, A, 1)         %<-- computes X times A in mode-1
%   Y = ttm(X, A.', 1, 't')   %<-- same as above
%   Y = ttm(X, @(x)(A*x), 1) %<-- same as above
%   Y = ttm(X, {A,B,C,D})            %<-- 4-way multiply
%   Y = ttm(X, {A,B,C,D}, [1 2 3 4]) %<-- same as above
%   Y = ttm(X, {A',B',C',D'}, 'h')   %<-- same as above
%   Y = ttm(X, {C,D}, [3 4])     %<-- X times C in mode-3 & D in mode-4
%   Y = ttm(X, {D,C}, [4 3])     %<-- same as above
%   Y = ttm(X, {[],[],C,D})      %<-- same as above
%   Y = ttm(X, {A,B,D}, [1 2 4])   %<-- 3-way multiply
%   Y = ttm(X, {A,B,C,D}, -3)      %<-- same as above
%
%   See also TTT, HTENSOR/TTM, HTENSOR/TTT.
%

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

% Check number of arguments
if(nargin < 2)
  error('Requires at least two arguments.');
end

if(~isnumeric(x))
  error('First argument must be a MATLAB multidimensional array.');
end

% Case of only one matrix: Put it into a cell array.
if (~isa(A, 'cell'))
  if(~isnumeric(A) && ~isa(A, 'function_handle'))
    error(['Second argument must be either a cell array, a' ...
	   ' MATLAB multidimensional array or a function handle.']);
  else
    A = {A};
  end
else
  if(~all(cellfun( ...
      @(x)(isnumeric(x) || isa(x, 'function_handle')), A)))
    error(['All elements of cell array A must be either ' ...
	   'MATLAB multidimensional arrays or function handles.']);
  end
end

% Default values of optional arguments
transposed = false;
ctransposed = false;

% Insert additional arguments (note: only the data type matters, not
% the position in the argument list). If there are, e.g., two
% character arguments, the second one will overwrite the first one.
for ii=1:numel(varargin)
  if(isa(varargin{ii}, 'char') && ~isempty(varargin{ii}))
    if(varargin{ii}(1) == 't')
      transposed = true;
    elseif(varargin{ii}(1) == 'h')
      ctransposed = true;
    else
      fprintf('ttm: %dth argument unknown, ignored.\n', ii+2);
    end
  elseif(isindexvector(varargin{ii}) || isindexvector(-varargin{ii}))
    dims = varargin{ii};
  else
    error('Invalid argument dims.');
  end
end

if(~exist('dims', 'var'))
  d = max(ndims(x), numel(A));
  if(d ~= numel(A))
    error(['Invalid argument: number of elements in A must correspond ' ...
	   'to the order of tensor X, when dims is not given.'])
  end
  dims = 1:d;
  
elseif(all(dims < 0))
  d = max(ndims(x), numel(A));
  if(d ~= numel(A))
    error(['Invalid argument: number of elements in A must correspond ' ...
	   'to the order of tensor X, when negative dims are given.']);
  end

  if(any(-dims > d))
    error('DIMS contains dimensions outside of the range 1:d.')
  end
  
  dims = setdiff(1:d, -dims);
  A = A(dims);
else
  d = max(ndims(x), max(dims));
  if(numel(A) ~= numel(dims))
    error('A must have the same number of elements as DIMS.');
  end
end

% Now calculate X x_dims{1} A{1} ... x_dims{end} A{end}

% Calculate size (including singleton dimensions)
sz = size(x);
sz(end+1:d) = 1;

% Loop over cell array A
for ii=1:length(dims)

  if( all(size(A{ii}) == 0) )
    continue;
  end
  
  % Matricize X
  if(ismember(1, dims(ii)) || isa(A{ii}, 'function_handle'))
    X = matricize(x, dims(ii));
    transX = false;
  else
    compl_dims = setdiff(1:d, dims(ii));
    X = matricize(x, compl_dims);
    transX = true;
  end
  
  % Apply matrix A{ii}
  if(isnumeric(A{ii}))
    if(transposed)
      if(size(A{ii}, 1) ~= sz(dims(ii)))
	error('Matrix dimensions must agree.')
      end
      if(~transX)
	X = A{ii}.'*X;
      else
	X = X*A{ii};
      end
    elseif(ctransposed)
      if(size(A{ii}, 1) ~= sz(dims(ii)))
	error('Matrix dimensions must agree.')
      end
      if(~transX)
	X = A{ii}'*X;
      else
	X = X*conj(A{ii});
      end
    else
      if(size(A{ii}, 2) ~= sz(dims(ii)))
	error('Matrix dimensions must agree.')
      end
      if(~transX)
	X = A{ii}*X;
      else
	X = X*A{ii}.';
      end
    end
  elseif(isa(A{ii}, 'function_handle'))
    X = A{ii}(X);
  else
    error(['A{ii} is of incompatible type; should be a matrix or' ...
	   ' function handle.'])
  end
  
  % Update size of X
  if(transX == false)
    sz(dims(ii)) = size(X, 1);
  else
    sz(dims(ii)) = size(X, 2);
  end

  % Dematricize X
  if(transX == false)
    x = dematricize(X, sz, dims(ii));
  else
    x = dematricize(X, sz, compl_dims);
  end  
  
end
