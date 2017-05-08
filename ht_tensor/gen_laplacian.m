function Lapl = gen_laplacian(d, A, M)
%GEN_LAPLACIAN Operator htensor for operator with Laplacian structure.
%
%  Lapl = GEN_LAPLACIAN(d, A, M) describes the operator
%
%    M{d} x       ...         x M{2} x A{1} 
%  + M{d} x    ...     x M{3} x A{2} x M{1} 
%  +                   ... 
%  + A{d} x M{d-1} x   ...    x M{2} x M{1}.
%
%  If A is a matrix instead of a cell array, we use 
%  A{1} = ... = A{d} = A.
%  The same applies for a matrix M. In addition, if no third
%  argument is given, the matrices M are set to the identity
%  matrix, M{ii} = eye(size(A{ii})).
%
%  The operator Lapl is represented as an htensor of size n_1^2 x
%  ... x n_d^2, by vectorising the matrices, and storing these vectors
%  in the (sparse) leaf matrices.
%
%  See also LAPLACE_CORE, APPLY_MAT_TO_VEC, APPLY_MAT_TO_VEC.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

% Check cell array A
if(iscell(A))
  n = cellfun('size', A, 1);
else
  n = size(A, 1)*ones(1, d);
  A = {A};
  for ii=2:d
    A{ii} = A{1};
  end
end

% Check or initialize cell array M
if(nargin == 2)
  M = cell(1, d);
  for ii=1:d
    M{ii} = speye(n(ii));
  end
elseif(~iscell(M))
  M = {M};
  for ii=2:d
    M{ii} = M{1};
  end
end

% Initialize htensor Lapl
Lapl = htensor(1:d);

U = cell(1, Lapl.nr_nodes);
B = cell(1, Lapl.nr_nodes);

% Core tensor (see LAPLACE_CORE.m)
B_ = dematricize([1 0; 0 1; 0 1; 0 0], [2 2 2], [1 2]);

% Compute U{ii}, B{ii}
for ii=2:Lapl.nr_nodes
  
  if( Lapl.is_leaf(ii) )
    dim = Lapl.dims{ii};
    U{ii} = [M{dim}(:), A{dim}(:)];
  else
    B{ii} = B_;
  end

end

B{1} = [0 1; 1 0];

% Construct htensor Lapl
Lapl = htensor(Lapl.children, Lapl.dims, U, B, false);