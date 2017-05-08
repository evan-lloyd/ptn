function fHandle = handle_inv_mat(A)
%HANDLE_INV_MAT Function handle that applies operator to htensor.
% 
%  fHandle = HANDLE_INV_MAT(A) returns a function handle,
%  which applies the operator
%
%  inv(A{d}) x ... x inv(A{1}),     if A is a cell array,
%
%  I x ... x I x inv(A),         if A is a matrix,
%
%  to an htensor.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

  if(isa(A, 'cell'))
    d = length(A);
    for ii=1:d
      A{ii} = @(x)(A{ii}\x);
    end
  else
    d = 1;
    A = @(x)(A\x);
  end

  fHandle = @apply_inv_mat;
  
  function Ax = apply_inv_mat(x, opts)
  
    Ax = ttm(x, A, 1:d);

  end

end