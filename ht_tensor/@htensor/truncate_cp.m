function [x, err, sv] = truncate_cp(cp, opts, weights)
%TRUNCATE_CP Truncates a CP tensor to a lower-rank htensor.
%
%   Y = TRUNCATE(CP, OPTS) truncates tensor CP defined by
%     SUM_i CP{d}(:, i) x ... CP{1}(:, i) 
%   to lower-rank htensor Y, depending on the options:
%
%   1) No rank k(ii) can be bigger than OPTS.MAX_RANK. 
%   2) The relative error in tensor norm cannot be bigger than
%   OPTS.REL_EPS, except when the first condition requires it.
%   3) The absolute error in tensor norm cannot be bigger than
%   OPTS.ABS_EPS, except when the first condition requires it.
%
%   Y = TRUNCATE(CP, OPTS, W) truncates the tensor defined by 
%     SUM_i W(i) CP{d}(:, i) x ... CP{1}(:, i) 
%   to lower-rank htensor Y. OPTS has the effects described above.
%
%   The dimension tree can be chosen by the optional field
%   OPTS.TREE_TYPE. For TREE_TYPE options, see HTENSOR.DEFINE_TREE.
%
%
%   The expected errors in each node and overall are displayed if
%   OPTS.PLOT_ERRTREE is set to true.
%
%   See also HTENSOR, ORTHOG, GRAMIANS
%

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

if(nargin == 1)
  error('Requires at least 2 arguments.')
end

if(nargin == 2)
  if(isa(cp, 'ktensor'))
    cp_U = cp.U;
    cp_weights = cp.lambda;
  elseif(isa(cp, 'cell'))
    cp_U = cp;
    cp_weights = ones(size(cp{1}, 2), 1);
  else
    error('Invalid arguments.');
  end
elseif(nargin == 3 && isa(cp, 'cell'))
  cp_U = cp;
  cp_weights = weights;
else
  error('Invalid arguments.');
end

if(size(cp_weights, 2) ~= 1)
  cp_weights = transpose(cp_weights);
end

if(~isa(opts, 'struct') || ~isfield(opts, 'max_rank') )
  error(['Second argument must be a MATLAB struct with field max_rank,' ...
	 ' and optionally fields abs_eps and/or rel_eps.']);
end

d = numel(cp_U);

% From opts.rel_eps/opts.abs_eps, calculate the error permitted for
% each truncation (there are 2*d - 2 truncations overall).
if(isfield(opts, 'rel_eps'))
  opts.rel_eps = opts.rel_eps/sqrt(2*d - 2);
end

if(isfield(opts, 'abs_eps'))
  opts.abs_eps = opts.abs_eps/sqrt(2*d - 2);
end

for ii=1:d
  
  % Calculate QR-decomposition
  [Q, R] = qr(cp_U{ii}, 0);
  
  % Make sure rank doesn't become zero
  if(size(R, 1) == 0)
    Q = ones(size(Q, 1), 1);
    R = ones(1, size(R, 2));
  end
  Q_cell{ii} = Q;
  R_cell{ii} = R;
  
  cp_U{ii} = R;
  
end

% Calculate the gramians of x
if(~isfield(opts, 'tree_type'))
  opts.tree_type = '';
end

G = htensor.gramians_cp(cp_U, cp_weights, opts.tree_type);

x = htensor(cp_U, cp_weights, opts.tree_type);

x.U(x.dim2ind) = Q_cell;

tmp = R_cell;
R_cell = cell(1, x.nr_nodes);
R_cell(x.dim2ind) = tmp;

% err represents the node-wise truncation errors
err = zeros(x.nr_nodes, 1);

% Go from leaves to root (though the order does not matter)
for ii=x.nr_nodes:-1:2

  if(~x.is_leaf(ii))
    
    ii_left  = x.children(ii, 1);
    ii_right = x.children(ii, 2);
    
    % Multiply Rleft, Rright to B, matricize B{ii}
    B_mat = khatrirao(R_cell{ii_right}, R_cell{ii_left});
    
    % Calculate QR-decomposition
    [Q, R] = qr(B_mat, 0);
    
    R_cell{ii} = R;
    
    % Calculate dimensions of "tensor" Q
    tsize_new = [size(R_cell{ii_left}, 1), ...
		 size(R_cell{ii_right}, 1), ...
		 size(Q, 2)];
    
    % Reshape Q to tensor B{ii}
    x.B{ii} = dematricize(Q, tsize_new, [1 2], 3);
    
  end
  
  % Update Gramian
  G{ii} = R_cell{ii}*G{ii}*R_cell{ii}';
  
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
  
  R_cell{ii} = U_'*R_cell{ii};
  
end

root_left  = x.children(1, 1);
root_right = x.children(1, 2);

% Multiply Rleft, Rright to B:
x.B{1} = ttm(x.B{1}, {R_cell{root_left}, R_cell{root_right}}, [1 2]);

x.is_orthog = true;

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
  err_ = err; err_(x.children(1, 1)) = 0;

  % Calculate upper bound and c from ||x - x_|| <= c ||x - x_best||
  ind_lvl = find(x.is_leaf);
  factor = sqrt(length(ind_lvl));
  err_bd    = norm(err_(ind_lvl));
  for ii=1:max(x.lvl)
    ind_lvl = find(x.lvl == ii & ~x.is_leaf);
    factor = factor + sqrt(length(ind_lvl));
    err_bd = err_bd + norm(err_(ind_lvl));
  end
  
  fprintf(['\nLower/Upper bound for best approximation error:\n' ...
	   '%e <= ||X - X_best|| <= ||X - X_|| <= %e\n'], ...
	  max(err_bd/factor, max(err)), err_bd);
end
