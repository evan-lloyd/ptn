function [x, err, sv] = truncate_std(x, opts)
%TRUNCATE_STD Truncates an htensor to a lower-rank htensor.
%
%   Y = TRUNCATE_STD(X, OPTS) truncates htensor X to lower-rank htensor
%   Y, depending on the options:
%
%   1) No rank k(ii) can be bigger than OPTS.MAX_RANK. 
%   2) The relative error in tensor norm cannot be bigger than
%   OPTS.REL_EPS, except when the first condition requires it.
%   3) The absolute error in tensor norm cannot be bigger than
%   OPTS.ABS_EPS, except when the first condition requires it.
%
%   The expected errors in each node and overall are displayed if
%   OPTS.PLOT_ERRTREE is set to true.
%
%   See also HTENSOR, ORTHOG, GRAMIANS

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

if(nargin ~= 2)
  error('Requires exactly 2 arguments.')
end

if(~isa(x, 'htensor'))
  error('First argument must be of class htensor.')
end

if(~isa(opts, 'struct') || ~isfield(opts, 'max_rank') )
  error(['Second argument must be a MATLAB struct with field max_rank,' ...
	 ' and optionally fields abs_eps and/or rel_eps.']);
end

% From opts.rel_eps/opts.abs_eps, calculate the error permitted for
% each truncation (there are 2*d - 2 truncations overall).
if(isfield(opts, 'rel_eps'))
  opts.rel_eps = opts.rel_eps/sqrt(2*ndims(x) - 2);
end

if(isfield(opts, 'abs_eps'))
opts.abs_eps = opts.abs_eps/sqrt(2*ndims(x) - 2);
end

% Orthogonalize x (does nothing if x is already orthogonal)
x = orthog(x);

% Calculate the gramians of x
G = gramians(x);

% err represents the node-wise truncation errors
err = zeros(x.nr_nodes, 1);

% Go from leaves to root (though the order does not matter)
for ii=x.nr_nodes:-1:2

  % calculate the left singular vectors U_ and singular values s
  % of X_{ii} from the gramian G{ii}.
  [U_, sv{ii}] = htensor.left_svd_gramian(G{ii});
  
  % Calculate rank k to use, and expected error.
  [k, err(ii)] = htensor.trunc_rank(sv{ii}, opts);
  
  % truncate U_
  U_ = U_(:, 1:k);
  
  % Apply U_ to node ii:
  if(x.is_leaf(ii))
    x.U{ii} = x.U{ii}*U_;
  else
    x.B{ii} = ttm(x.B{ii}, U_, 3, 't');
  end
  
  % Parent node
  ii_par = x.parent(ii);
  
  % Apply U_ to parent node
  if(x.is_left(ii))
    x.B{ii_par} = ttm(x.B{ii_par}, U_, 1, 'h');
  else
    x.B{ii_par} = ttm(x.B{ii_par}, U_, 2, 'h');
  end
  
end

x.is_orthog = false;

% Display the expected error in each node and overall.
if(isfield(opts, 'plot_errtree') && opts.plot_errtree == true)
  disp_tree(x, 'truncation_error', err);
  
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
