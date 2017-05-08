function z = times(x, y)
%TIMES Element-by-element multiplication of htensors.
%
%   Z = TIMES(X, Y) Elementwise multiplication of htensors X
%   and Y. They must have identical dimension trees and the sizes.
%
%   This operation is expensive, as the ranks of the resulting
%   htensor Z are
%        Z.rank(ii) = X.rank(ii)*Y.rank(ii)
%
%   See also HTENSOR/ELEM_MULT.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

% Check compatibility of dimension trees.
if(~equal_dimtree(x, y))
  error('Dimension trees of X and Y differ.');
end

% Check sizes
if(~isequal(size(x), size(y)))
  error('Tensor dimensions must agree.')
end

z = x;

for ii=1:z.nr_nodes
  
  if(z.is_leaf(ii))
    z.U{ii} = khatrirao_t(x.U{ii}, y.U{ii});
  else
    sz_x = size(x.B{ii});
    sz_x(end+1:3) = 1;
    sz_y = size(y.B{ii});
    sz_y(end+1:3) = 1;

    % Calculate "3-D Kronecker product"
    z.B{ii} = zeros(sz_x.*sz_y);
    for jj=1:size(x.B{ii}, 3)
      for kk=1:size(y.B{ii}, 3)      
	z.B{ii}(:, :, kk+(jj-1)*size(y.B{ii}, 3)) = ...
	    kron(x.B{ii}(:, :, jj), y.B{ii}(:, :, kk));
      end
    end
  end
  
end

z.is_orthog = false;