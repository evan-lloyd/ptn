function fHandle = handle_lin_mat(A, alpha)
%HANDLE_LIN_MAT Function handle that applies operator to htensor.
% 
%  fHandle = HANDLE_LIN_MAT(A, ALPHA) returns a function handle,
%  which applies the operator
%
%  I x ... x I x A{1} + I x ... x D{2} x A{2} + ... 
%                           ... + D{d} x I x ... x I x A{d}
%
%  to an htensor, truncating after every addition.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

  d = length(A);
  m = cellfun('length', alpha);
  
  fHandle = @apply_lin_mat;
  
  function Ax = apply_lin_mat(x, opts)
  
    Ax = ttm(x, A{1}, 1);
    %norms(1) = norm(Ax);
    for jj=2:d
      Ajj_x = ttm(x, A{jj}, 1);
      Ajj_x = ttm(Ajj_x, spdiags(alpha{jj}', 0, m(jj), m(jj)), jj);
      Ax = Ax + Ajj_x;
      %norms(jj) = norm(Ajj_x);
      
      if(nargin == 2)
	Ax = truncate(Ax, opts);
      end
    end
    %norms
    
  end

end

