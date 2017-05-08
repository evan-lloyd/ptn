function [ht, err, sv] = truncate_rtl(x, opts)
%TRUNCATE_RTL Truncate a tensor to an htensor, Root-to-Leaves method.
%
%   Y = CALC_DECOMPOSITION_RTL(X, OPTS) truncates tensor X to
%   htensor Y, depending on the options:
%
%   1) No rank k(ii) cannot be bigger than OPTS.MAX_RANK. 
%   2) The relative error in tensor norm cannot be bigger than
%   OPTS.REL_EPS, except when the first condition requires it.
%   3) The absolute error in tensor norm cannot be bigger than
%   OPTS.ABS_EPS, except when the first condition requires it.
%
%   The expected errors in each node and overall are displayed if
%   OPTS.PLOT_ERRTREE is set to true.
%
%   *** This method is only provided for illustration, ***
%   *** TRUNCATE_LTR is always faster, and has the     ***
%   *** same error bound.                              ***
%
%   The dimension tree can be chosen by the optional field
%   OPTS.TREE_TYPE. For TREE_TYPE options, see HTENSOR.DEFINE_TREE.
%
%   Calculation of the left singular vectors and values is
%   controlled by OPTS.SV:
%   OPTS.SV = 'svd': Applying qr and svd to X_{ii} (default)
%   OPTS.SV = 'gramian': Applying eig to X_{ii}*X_{ii}'
%
%   See also HTENSOR, TRUNCATE, TRUNCATE_LTR
%

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

if(nargin ~= 2)
  error('Requires exactly 2 arguments.')
end

if(~isnumeric(x))
  error('First argument must be a numeric (multidimensional) array.')
end

if(~isa(opts, 'struct') || ~isfield(opts, 'max_rank') )
  error(['Second argument must be a MATLAB struct with field max_rank,' ...
	 ' and optionally abs_eps and/or rel_eps.']);
end

% Initialize htensor t
if(isfield(opts, 'tree_type'))
  ht = htensor(size(x), opts.tree_type);
else
  ht = htensor(size(x));
end

% initialize cells of matrices U and B:
U = cell(1, ht.nr_nodes);
B = cell(1, ht.nr_nodes);

% err represents the node-wise truncation errors
err = zeros(1, ht.nr_nodes);

% From opts.rel_eps/opts.abs_eps, calculate the error permitted for
% each truncation (there are 2*d - 2 truncations overall).
if(isfield(opts, 'rel_eps'))
opts.rel_eps = opts.rel_eps/sqrt(2*ndims(x) - 2);
end

if(isfield(opts, 'abs_eps'))
opts.abs_eps = opts.abs_eps/sqrt(2*ndims(x) - 2);
end

% Traverse tree from leafs to root
for ii=ht.nr_nodes:-1:1
  
  % Matricization of x corresponding to node ii
  x_mat = matricize(x, ht.dims{ii});
  
  % special case of root node: matricization is a vector
  if(ii == 1)
    U{ii} = x_mat;
    k(ii) = 1;
  else
    
    % Calculate left singular vectors U_ and singular values of x_mat
    if(~isfield(opts, 'sv'))
      opts.sv = 'svd';
    end
    
    if(strcmp(opts.sv, 'gramian'))
      [U_, sv{ii}] = htensor.left_svd_gramian(x_mat*x_mat');
    elseif(strcmp(opts.sv, 'svd'))
      [U_, sv{ii}] = htensor.left_svd_qr(x_mat);
    else
      error('Invalid value of OPTS.SV.');
    end
    
    % Calculate rank k to use, and expected error
    [k(ii), err(ii)] = htensor.trunc_rank(sv{ii}, opts);
    
    % Save left singular vectors U for later
    U{ii} = U_(:, 1:k(ii));
  
  end
  
  % Calculate tensor B from U and the Us of the child nodes
  if(~ht.is_leaf(ii))
    
    % Child nodes' indexes
    ii_left  = ht.children(ii, 1);
    ii_right = ht.children(ii, 2);
    
    % Number of rows of child node matrices
    n_left = size(U{ii_left}, 1);
    n_right = size(U{ii_right}, 1);
    
    % reshape U{ii} to a k(ii) x n_left x n_right tensor
    U_tensor = dematricize(U{ii}, [n_left, n_right, k(ii)], ...
			   [1 2], 3);
    
    % Apply matrices U{ii_left} and U{ii_right} to find
    % transfer tensor B{ii}
    B{ii} = ttm(U_tensor, {U{ii_left}, U{ii_right}}, ...
		     [1 2], 'h');
    
    % Old version
    %B{ii} = zeros([k(ii_left), k(ii_right), k(ii)]);
    %for jj=1:k(ii)
    %  U_mat = reshape(U{ii}(:, jj), size(U{ii_left}, 1), ...
    % 		      size(U{ii_right}, 1));
    %  B{ii}(:, :, jj) = U{ii_left}'*U_mat*U{ii_right};
    %end
    
    % Free unused storage
    if(~ht.is_leaf(ii_left))
      U{ii_left} = [];
    end
    if(~ht.is_leaf(ii_right))
      U{ii_right} = [];
    end
  end
  
end

% Free unused storage: 
U{1} = [];

% Call htensor constructor
ht = htensor(ht.children, ht.dims, U, B, false);

% Display the expected error in each node and overall.
if(isfield(opts, 'plot_errtree') && opts.plot_errtree == true)
  disp_tree(ht, 'truncation_error', err);
  
  % We know from theory that
  %
  % ||X - X_best|| <= ||X - X_|| <= err_bd <= factor*||X - X_best||
  %
  % and max(err) <= ||X - X_best||, therefore
  %
  % max(err_bd/factor, max(err)) <= ||X - X_best|| <= ||X - X_|| <= err_bd
  %
  % give upper and lower bounds for the best approximation as well
  % as the truncated version constructed here.
  %
  
  % Count top-level truncation only once
  err_ = err; err_(ht.children(1, 1)) = 0;
  
  % Calculate upper bound and factor c from ||x - x_|| <= c ||x - x_best||
  err_bd = norm(err_); factor = sqrt(2*ndims(ht)-3);
  
  fprintf(['\nLower/Upper bound for best approximation error:\n' ...
	   '%e <= ||X - X_best|| <= ||X - X_|| <= %e\n'], ...
	  max(err_bd/factor, max(err)), err_bd);
end
