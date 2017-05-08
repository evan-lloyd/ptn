function [ht, err, sv] = truncate_ltr(x, opts)
%TRUNCATE_LTR Truncate a tensor to an htensor, Leaves-to-Root method.
%
%   Y = TRUNCATE_LTR(X, OPTS) truncates tensor X to
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
%   The dimension tree can be chosen by the optional field
%   OPTS.TREE_TYPE. For TREE_TYPE options, see HTENSOR.DEFINE_TREE.
%
%   Calculation of the left singular vectors and values is
%   controlled by OPTS.SV:
%   OPTS.SV = 'svd': Using qr and svd on X_{ii} (default)
%   OPTS.SV = 'gramian': Using eig on X_{ii}*X_{ii}'
%
%   See also HTENSOR, TRUNCATE, TRUNCATE_RTL
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
	 ' and optionally fields abs_eps and/or rel_eps.']);
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

% Make a temporary copy of tensor x
x_ = full(x);

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

% Traverse leafs of the dimension tree
for ii=find(ht.is_leaf)
  
  % Matricization of x corresponding to node ii
  x_mat = matricize(x, ht.dims{ii});
  
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
  
  % Calculate rank k to use, and expected error.
  [k(ii), err(ii)] = htensor.trunc_rank(sv{ii}, opts);
  
  % Save left singular vectors U for later
  U{ii} = U_(:, 1:k(ii));
  
  % Reduce tensor x_ along this dimension
  x_ = ttm(x_, U{ii}, ht.dims{ii}, 'h');
  
end

% Set x to be the reduced tensor x_
x = x_;

% Go through levels from leafs to root node
for lvl_iter = max(ht.lvl):-1:0
  
  % Go through all nodes at given level
  for ii=find(ht.lvl == lvl_iter)
    
    % Leafs have already been treated, we skip this
    if(ht.is_leaf(ii))
      continue;
    end
    
    % Matricization of x corresponding to node ii
    x_mat = matricize(x, ht.dims{ii});
    
    % special case of root node: matricization is a vector
    if(ii == 1)   
      U_ = x_mat;
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
	error('Invalid argument OPTS.SV.');
      end
      
      % Calculate rank k to use, and expected error.
      [k(ii), err(ii)] = htensor.trunc_rank(sv{ii}, opts);
      
      % Cut U_ after first k columns
      U_ = U_(:, 1:k(ii));
    end
    
    % Child nodes' indexes
    ii_left  = ht.children(ii, 1);
    ii_right = ht.children(ii, 2);
    
    % reshape B{ii} from matrix U_ to a 
    % k(ii) x k(ii_left) x k(ii_right) tensor, 
    B{ii} = dematricize(U_, [k(ii_left), k(ii_right), k(ii)], ...
			[1 2], 3);
			  
    % Reduce tensor x_ along dimensions x.dims{ii}; this will
    % change the number of dimensions of x_:
    
    % Matricization of x_, making dims{ii} the row dimensions
    x_mat_ = matricize(x_, ht.dims{ii});
    
    % calculate B{ii}'*x_mat_
    U_x_mat = U_'*x_mat_;
    
    % Instead of eliminating one of the dimensions, just set
    % it to be a singleton, to keep the dimension order consistent
    tsize_red = size(x_);
    tsize_red(ht.dims{ii_left }(1)) = k(ii);
    tsize_red(ht.dims{ii_right}(1)) = 1;
    
    % Reshape x_mat_ to tensor x_
    x_ = dematricize(U_x_mat, tsize_red, ht.dims{ii});
    
  end
  
  % Set x to be the reduced tensor x_  
  x = x_;
end

% Call htensor constructor
ht = htensor(ht.children, ht.dims, U, B, true);

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
