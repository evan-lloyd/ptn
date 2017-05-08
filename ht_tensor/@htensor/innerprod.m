function s = innerprod(x1, x2)
%INNERPROD Inner product of two htensors.
%
%   S = INNERPROD(X1,X2) computes the inner product between two
%   htensors X1 and X2. The htensors must have the identical dimension
%   trees and sizes.
%
%   For complex numbers, the complex conjugate of X1 is used.
%
%   See also HTENSOR

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

if(~isa(x1, 'htensor') || ~isa(x2, 'htensor'))
  error('Both arguments X1 and X2 must be of class htensor.')
end

% Check sizes
if(~isequal(size(x1), size(x2)))
  error('Tensor dimensions must agree.')
end

M = cell(x1.nr_nodes, 1);

% Start at leaves, move up the levels
for ii=x1.nr_nodes:-1:1
  
  jj = find(cellfun(@(y)(isequal(sort(x1.dims{ii}), sort(y))), ...
		    x2.dims));
  
  if(~isscalar(jj))
    error('Dimension trees of X1 and X2 are incompatible.');
  end
  
  if(x1.is_leaf(ii))
    % M_t = U1_t' * U2_t
    M{jj} = full(x1.U{ii}'*x2.U{jj});
  else
    jj_left  = x2.children(jj, 1);
    jj_right = x2.children(jj, 2);
    
    ii_left  = x1.children(ii, 1);
    
    % M_t = B1_t' * (M_t1 kron M_t2) * B2_t
    % (interpreting B1_t, B2_t to be in matricized form)
    B_ = ttm(x2.B{jj}, { M{jj_left}, M{jj_right} }, [1 2]);
    
    % Determine whether there is a crossover between x1 and x2
    if(isequal(sort(x1.dims{ii_left}), ...
	       sort(x2.dims{jj_left})) )
      M{jj} = ttt(x1.B{ii}, B_, [1 2]);
    else
      M{jj} = ttt(x1.B{ii}, B_, [1 2], [2 1]);
    end
    
    % If there is a singleton dimension in x1, M{jj} becomes a
    % column vector, but should be a row vector:      
    if(size(x1.B{ii}, 3) == 1)
      M{jj} = M{jj}.';
    end
    
    % Save memory
    M{jj_left} = []; M{jj_right} = [];
  end
end

s = M{1};
