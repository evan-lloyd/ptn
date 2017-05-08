function x = plus(x1, x2)
%PLUS Binary addition for htensor.
%
%   X = PLUS(X1, X2) adds two htensors of the same size.
%   The result is an htensor of the same size.
%
%   See also HTENSOR.
%

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

% Check type of x1, x2
if( ~isa(x1, 'htensor') || ~isa(x2, 'htensor') )
  error('X1 and X2 must be of class htensor.');
end

% Check compatibility of dimension trees.
if(~equal_dimtree(x1, x2))
  error('Dimension trees of X1 and X2 differ.');
end

% Check sizes
if(~isequal(size(x1), size(x2)))
  error('Tensor dimensions must agree.')
end

% Set properties of result tensor t
x = x1;
rank = x1.rank + x2.rank;
rank(1) = 1;

x.is_orthog = false;

% Construct new leaf matrices
for ii=find(x.is_leaf)
  x.U{ii} = [x1.U{ii}, x2.U{ii}];
end

% Construct new node tensors
for ii=find(~x.is_leaf)
  
  ii_left  = x.children(ii, 1);
  ii_right = x.children(ii, 2);
  
  % Allocate space
  x.B{ii} = zeros([rank(ii_left), rank(ii_right), rank(ii)]);
  
  % Special treatment is necessary at root node
  if(ii ~= 1)
    
    x.B{ii}(1:x1.rank(ii_left), 1:x1.rank(ii_right), 1:x1.rank(ii)) = ...
	x1.B{ii};
    x.B{ii}(x1.rank(ii_left)+1:end, x1.rank(ii_right)+1:end, ...
	    x1.rank(ii)+1:end) = x2.B{ii};
  else
    
    x.B{ii}(1:x1.rank(ii_left)    , 1:x1.rank(ii_right)    ) = x1.B{ii};
    x.B{ii}(x1.rank(ii_left)+1:end, x1.rank(ii_right)+1:end) = x2.B{ii};
    
  end
end
