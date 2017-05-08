function x = htenones(sz, tree_type)
%HTENONES Tensor with all elements equal 1.
%
%   X = HTENONES(SZ) returns the rank one htensor of size SZ, with
%   all elements equal to 1.
%
%   X = HTENONES(SZ, TREE_TYPE) returns the rank one htensor of size
%   SZ, with all elements equal to 1, and the dimension tree
%   controlled by TREE_TYPE.
%
%   See also HTENSOR, HTENRAND.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

if(nargin < 1)
  error('Requires at least 1 argument.');
elseif(nargin == 1)
  tree_type = '';
end

% Check sz
if( ~isindexvector(sz) )
  error('SZ must be a vector of positive integers.');
end

% Check tree_type
if(~ischar(tree_type))
  error('Third argument TREE_TYPE must be a char array.');
end

% Construct htensor x with zero entries
x = htensor(sz, tree_type);

for ii=1:x.nr_nodes
  
    if(x.is_leaf(ii))
      
      dim_ii = x.dims{ii};

      % Initialize U{ii}
      U{ii} = ones(sz(dim_ii), 1);
      
    else
      
      % Construct random tensor B{ii}
      B{ii} = 1;
		
    end
end

x = htensor(x.children, x.dims, U, B);
