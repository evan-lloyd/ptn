function [sin_htd, cos_htd] = gen_sin_cos(t, tree_type)
%GEN_SIN_COS Generates htensors sin(t1+...+td), cos(t1+...+td).
%
%  [SIN_HTD, COS_HTD] = GEN_SIN_COS(T) generates htensors
%
%  SIN_HTD(i1,..,id) = sin(T{1}(i1) + ... + T{d}(id)),
%  COS_HTD(i1,..,id) = cos(T{1}(i1) + ... + T{d}(id)),
%
%  that have hierarchical ranks equal to 2.
%
%  [SIN_HTD, COS_HTD] = GEN_SIN_COS(T, TREE_TYPE) allows a choice on
%  the dimension tree of the results. For TREE_TYPE options, see
%  HTENSOR.DEFINE_TREE.
%
%
%  See also LAPLACE_CORE.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

% Check t
if(~all(cellfun(@isvector, t)))
  error('All entries of T must be numeric vectors');
end

d = numel(t);
sz = cellfun('length', t);

for ii=1:d
  if(size(t{ii}, 1) == 1 && size(t{ii}, 2) > 1)
    t{ii} = t{ii}';
  end
end

if(nargin == 1)
  tree_type = '';
end

% Check tree_type
if(~ischar(tree_type))
  error('Second argument tree_type must be a char array.');
end

% Initialize htensor x
x = htensor(sz, tree_type);

% Compute U{ii}, B{ii}
U = x.U;
B = x.B;

for ii=2:x.nr_nodes
  if(x.is_leaf(ii))
    dim_ii = x.dims{ii};
    U{ii} = [sin(t{dim_ii}), cos(t{dim_ii})];
  else
    B{ii} = dematricize([0 -1; 1 0; 1 0; 0 1], [2 2 2], [1 2], 3);
  end
end

% Calculate sin_htd
B{1} = [0 1; 1 0];
sin_htd = htensor(x.children, x.dims, U, B);

% Calculate cos_htd
B{1} = [-1 0; 0 1];
cos_htd = htensor(x.children, x.dims, U, B);
