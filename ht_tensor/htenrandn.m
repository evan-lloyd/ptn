function x = htenrandn(sz, orthog, tree_type, k)
%HTENRANDN Random htensor.
%
% The entries of U_t, B_t are normally distributed pseudo-random numbers.
%
%   X = HTENRANDN(SZ) forms a balanced htensor of size SZ, with
%   ranks randomly distributed between 3 and 6.
%
%   X = HTENRANDN(SZ, ORTHOG) forms an htensor of size SZ, with ranks
%   randomly distributed between 3 and 6. If ORTHOG='orthog', the
%   htensor is orthogonal.
%
%   X = HTENRANDN(SZ, ORTHOG, TREE_TYPE) forms an htensor of size SZ,
%   where the dimension tree is controlled by TREE_TYPE, with ranks
%   randomly distributed between 3 and 6.
%
%   X = HTENRANDN(SZ, ORTHOG, TREE_TYPE, K) forms an htensor of size
%   SZ, where the dimension tree is controlled by TREE_TYPE, and the
%   ranks are given by the (2*dim-1)-vector K.
%
%   See also HTENSOR, RANDN, (Tensor Toolbox TENRAND, SPTENRAND).

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

if(nargin < 1)
  error('Requires at least 1 argument.');
elseif(nargin == 1)
  orthog = '';
  tree_type = '';
elseif(nargin == 2)
  tree_type = '';
end

% Check sz
if( ~isindexvector(sz) )
  error('SZ must be a vector of positive integers.');
end

% Check orthog
if(~ischar(orthog))
  error('Second argument ORTHOG must be a char array.');
end

% Check tree_type
if(~ischar(tree_type))
  error('Third argument TREE_TYPE must be a char array.');
end

% Construct htensor x with zero entries
x = htensor(sz, tree_type);

% Random k if none is assigned
if(nargin <= 3)
  k = floor(4*rand(x.nr_nodes, 1))+3;
end

% Check k
if( ~isindexvector(k) || length(k) ~= x.nr_nodes)
  error('K must be a vector of 2*d-1 positive integers.');
end

% Make sure that the rank in dimension ii is not greater than the
% size in dimension ii.
for ii=1:ndims(x)
  if( k(x.dim2ind(ii)) > size(x, ii) )
    k(x.dim2ind(ii)) = size(x, ii);
  end
end

% Set root rank to 1
k(1) = 1;

for ii=1:x.nr_nodes
  
    if(x.is_leaf(ii))
      
      if(strcmp(orthog, 'orthog'))
	% Initialize orthogonal U{ii}
	U{ii} = orth(randn(x.size(x.dims{ii}), k(ii)));
      else
	% Initialize non-orthogonal U{ii}
	U{ii} = randn(x.size(x.dims{ii}), k(ii));
      end
      
    else
        ii_left  = x.children(ii, 1);
        ii_right = x.children(ii, 2);
        
	% Construct random tensor B{ii}
        B{ii} = randn([k(ii_left), k(ii_right), k(ii)]);
	
	if(strcmp(orthog, 'orthog'))
	  % Matricize B{ii}
	  B_mat = matricize(B{ii}, [1 2], 3);
	  
	  % Calculate orthonormal basis
	  Q_mat = orth(B_mat);
	  
	  % Reshape Q_mat to 3d-array B{ii}
	  B{ii} = dematricize(Q_mat, size(B{ii}), [1 2], 3);
	end
	
    end
end

x = htensor(x.children, x.dims, U, B, strcmp(orthog, 'orthog'));
