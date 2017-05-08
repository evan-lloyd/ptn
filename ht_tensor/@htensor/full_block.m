function y = full_block(x, index)
%FULL_BLOCK Return block of htensor as a (dense) tensor.
%
%   Y = FULL_BLOCK(X, INDEX) converts a block of htensor X to a
%   (dense) tensor Y. The Dx2-matrix INDEX gives the start and end
%   index in each dimension.
%
%   Examples:
%   x = htenrandn([2 4 3 5]);
%   y = full_block(x, [1 2; 1 1; 2 3; 1 5])
%   z = full(x(1:2, 1, 2:3, :));   % equal to y
%
%   See also HTENSOR, TENSOR, HTENSOR/FULL, HTENSOR/SUBSREF.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

% Check x
if(~isa(x, 'htensor'))
  error('First argument must be of class htensor.');
end

% Check index
if(~isnumeric(index) || any(size(index) ~= [ndims(x), 2]) || ...
   any(ceil(index(:)) ~= floor(index(:))) || any(index(:) <= 0) )
  error(['Second argument INDEX must be a Dx2-matrix of indexes,' ...
	 ' where D = ndims(X).']);
end

% Go through all leafs
for ii=find(x.is_leaf)
  % Restrict leaf matrix to the elements given by index
  x.U{ii} = x.U{ii}(index(x.dims{ii}, 1):index(x.dims{ii}, 2), :);
end

% convert x to dense tensor y
y = full(x);
