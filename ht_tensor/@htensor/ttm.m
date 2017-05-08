function x = ttm(x, A, varargin)
%TTM Tensor-times-matrix for htensor.
%
%   Y = TTM(X, A, N) computes the tensor-times-matrix product in 
%   mode N,
%     Y = X x_N A.
%   The arguments must satisfy size(X, N) == size(A, 2), and the
%   result Y will have size(Y, N) == size(A, 1).
%
%   Y = TTM(X, A) where A is a cell array with D = ndims(X)
%   entries, computes
%     Y = X x_1 A{1} x_2 A{2} ... x_D A{D}
%   If any entries of A are empty, nothing is done in that
%   dimension.
%
%   Y = TTM(X, A, EXCLUDE_DIMS) where A is a cell array with D =
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
%   Note that the usage is different from the function ttm in the
%   Tensor Toolbox.
%
%   Examples
%   X = htenrandn([5,3,4,2]);
%   A = rand(4,5); B = rand(4,3); C = rand(3,4); D = rand(3,2); 
%   Q = orth(rand(4, 4));
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
%   X = orthog(X);           %<-- orthogonalize htensor X
%   Y = ttm(X, Q, 3)         %<-- sets flag IS_ORTHOG to false
%   Y = ttm(X, Q, 3, 'o')    %<-- keeps flag IS_ORTHOG as true
%
%
%   See also TTM, HTENSOR/TTT, TTT.
%

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

% Check x
if(~isa(x, 'htensor'))
  error('First argument X must be of class htensor.');
end

% Check A
% Case of only one matrix: Put it into a cell array.
if (~isa(A, 'cell'))
  if( ~(isnumeric(A) && ndims(A) == 2) && ~isa(A, 'function_handle'))
    error(['Second argument must be either a cell array, a' ...
	   ' matrix or a function handle.']);
  else
    A = {A};
  end
else
  if(~all(cellfun( ...
      @(x)((isnumeric(x) && ndims(x) == 2) || isa(x, 'function_handle')), A)))
    error(['All elements of cell array A must be either ' ...
	   'matrices or function handles.']);
  end
end

% Default values of optional arguments
transposed = false;
ctransposed = false;
mat_orthog = false;

% Insert additional arguments (note: only the data type matters,
% not the position in the arguments list). If there are two char
% arguments, e.g., the second one is assigned to transposed.
if(nargin >= 3)
  for ii=1:nargin-2
    if(isa(varargin{ii}, 'char') && ~isempty(varargin{ii}))
      if(varargin{ii}(1) == 't')
	transposed = true;
      elseif(varargin{ii}(1) == 'h')
	ctransposed = true;
      elseif(varargin{ii}(1) == 'o')
	mat_orthog = true;
      else
	fprintf('ttm: %dth argument unknown, ignored.\n', ii+2);
      end
    elseif(isindexvector(varargin{ii}) || isindexvector(-varargin{ii}))
      dims = varargin{ii};
    else
      error('Invalid argument DIMS.');
    end
  end
end

% If dims is negative, select all dimensions except those indicated
if(~exist('dims', 'var'))
  
  if(ndims(x) ~= numel(A))
    error(['Invalid argument: number of elements in A must correspond ' ...
	   'to the order of tensor X, when DIMS is not given.'])
  end
  dims = 1:ndims(x);
  
elseif( all(dims < 0) )
  if(ndims(x) ~= numel(A))
    error(['Invalid argument: number of elements in A must correspond ' ...
	   'to the order of tensor X, when negative DIMS are' ...
	    ' given.']);
  end
  
  if(any(-dims > ndims(x)))
    error('DIMS contains dimensions outside of the range 1:d.')
  end
  
  dims = setdiff(1:ndims(x), -dims);

  A = A(dims);
else
  if(numel(A) ~= numel(dims))
    error('A must have the same number of elements as DIMS.');
  end
end

% Now calculate X x_dims{1} A{1} ... x_dims{end} A{end}

% Loop over cell array A
for ii=1:length(dims)
  
  if( all(size(A{ii}) == 0) )
    continue;
  end
  
  % Find node corresponding to dimension dims(ii)
  ind = x.dim2ind(dims(ii));
  
  if(isnumeric(A{ii}))
    if(transposed)
      if(size(A{ii}, 1) ~= size(x, dims(ii)))
	error('Matrix dimensions must agree.')
      end
      x.U{ind} = A{ii}.'*x.U{ind};
    elseif(ctransposed)
      if(size(A{ii}, 1) ~= size(x, dims(ii)))
	error('Matrix dimensions must agree.')
      end
      x.U{ind} = A{ii}'*x.U{ind};
    else
      if(size(A{ii}, 2) ~= size(x, dims(ii)))
	error('Matrix dimensions must agree.')
      end
      x.U{ind} = A{ii}*x.U{ind};
    end
  else % A{ii} is a function handle
    x.U{ind} = A{ii}(x.U{ind});
  end
  
end

x.is_orthog = x.is_orthog & mat_orthog;
