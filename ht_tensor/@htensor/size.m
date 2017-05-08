function sz = size(x, idx)
%SIZE Dimensions of htensor.
%  
%   D = SIZE(X) returns the sizes of each mode of tensor X in a
%   vector D with ndims(X) elements.
%
%   I = size(X,DIM) returns the size of the mode specified by the
%   scalar DIM.
%
%   Examples
%   x = rand(3,4,2,1); y = htenrandn([3 4 2 1]);
%   size(x) %<-- returns a length-3 vector
%   size(y) %<-- returns a length-4 vector
%   size(x,2) %<-- returns 4
%   size(y,2) %<-- same
%   size(x,5) %<-- returns 1
%   size(y,5) %<-- ERROR!
%
%   See also TENSOR, TENSOR/NDIMS, SIZE.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

if(~isa(x, 'htensor'))
  error('First argument must be of class htensor.');
end

sz = cellfun('size', x.U(x.dim2ind), 1);

if(nargin > 1)
  if(isindexvector(idx) && isscalar(idx) && idx <= numel(sz))
    sz = sz(idx);
  else
    error(['Second argument IDX must be an integer between 1 and' ...
	   ' %d.'], numel(sz));

  end
end
