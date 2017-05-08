function x = laplace_core(d, tree_type)
%LAPLACE_CORE Generates core tensor for Laplace operator.
%
%  X = LAPLACE_CORE(D) generates a 2 x ... x 2 htensor, which is
%  zero everywhere except for the elements
%
%  X(2, 1, ..., 1) = X(1, 2, 1, ..., 1) = ... = X(1, ..., 1, 2), 
%
%  which are equal to one. This is the core of the Laplace
%  operator, and more generally, of any tensor of the form
%  
%  a_1 x b_2 x ... x b_d  +  b_1 x a_2 x ... x b_d  +  ... 
%                                ...  +  b_1 x ... x b_(d-1) x a_d.
%
%  X = LAPLACE_CORE(D, TREE_TYPE) generates the same tensor, where
%  TREE_TYPE determines the dimension tree of X. For TREE_TYPE
%  options, see HTENSOR.DEFINE_TREE.
%
%  Example (x(i1, i2, i3, i4) = i1 + i2 + i3 + i4):
%  c = laplace_core(4);
%  U = [ones(100, 1), (1:100)'];
%  x = ttm(c, {U, U, U, U});
%
%  See also GEN_LAPLACIAN.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

if(nargin == 1)
  tree_type = '';
end

% Check tree_type
if(~ischar(tree_type))
  error('Second argument tree_type must be a char array.');
end

% Construct htensor x
x = htensor(2*ones(d, 1), tree_type);

% Construct U{ii}, B{ii}:
U = cell(1, x.nr_nodes);
B = cell(1, x.nr_nodes);

B_ = dematricize([1 0; 0 1; 0 1; 0 0], [2 2 2], [1 2]);

for ii=2:x.nr_nodes
  
  if( x.is_leaf(ii) )
    U{ii} = eye(2);    
  else
    B{ii} = B_;
  end

end
B{1} = [0 1; 1 0];

x = htensor(x.children, x.dims, U, B, true);