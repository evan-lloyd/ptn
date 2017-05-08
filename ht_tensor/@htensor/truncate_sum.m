function [x, err, sv] = truncate_sum(x_cell, opts)
%TRUNCATE_SUM Truncates a sum of htensors to a low-rank htensor.
%
%   Y = TRUNCATE(X, OPTS) truncates htensor X to lower-rank htensor
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

if(~isa(x_cell, 'cell') || ~all(cellfun( ...
      @(x)(isa(x, 'htensor')), x_cell)))
  error('First argument must be a cell array of htensors.')
end

% Check that size and dimension tree of all elements of x_cell are equal.
if( ~all(cellfun(@(foo)(isequal(size(foo), ...
				size(x_cell{1}))), x_cell)) ...
    || ~all(cellfun(@(foo)(equal_dimtree(foo, ...
					 x_cell{1})), x_cell)) )
  error('All htensors must have equal dimensions and dimension trees.');
end


if(~isa(opts, 'struct') || ~isfield(opts, 'max_rank') )
  error(['Second argument must be a MATLAB struct with field max_rank,' ...
	 ' and optionally fields abs_eps and/or rel_eps.']);
end

s = numel(x_cell);

x = x_cell{1};
for ii=1:x.nr_nodes
  x.U{ii} = [];
  x.B{ii} = [];
end

% From opts.rel_eps/opts.abs_eps, calculate the error permitted for
% each truncation (there are 2d-2 truncations overall).
if(isfield(opts, 'rel_eps'))
opts.rel_eps = opts.rel_eps/sqrt(2*ndims(x) - 2);
end

if(isfield(opts, 'abs_eps'))
opts.abs_eps = opts.abs_eps/sqrt(2*ndims(x) - 2);
end

for ii=find(x.is_leaf)
  
  % concatenate x_cell{:}.U{ii}
  x.U{ii} = cell2mat(cellfun(@(t)(t.U{ii}), x_cell, 'UniformOutput', false));
  
  % Calculate QR-decomposition
  [Q, R] = qr(x.U{ii}, 0);
  
  % Make sure rank doesn't become zero
  if(size(R, 1) == 0)
    Q = ones(size(Q, 1), 1);
    R = ones(1, size(R, 2));
  end
  
  Q_cell{ii} = Q;
  R_cell{ii} = R;
  
  x.U{ii} = R;
  
end

% Calculate the gramians of x
G = htensor.gramians_sum(x_cell);

% err represents the node-wise truncation errors
err = zeros(x.nr_nodes, 1);

% Rank structure of the summands
% x_cell{ii}.rank(jj) = blk_rank(jj, ii)
blk_rank = cell2mat(cellfun(@(t)(t.rank'), x_cell, 'UniformOutput', false));

% Go from leaves to root (though the order does not matter)
for ii=x.nr_nodes:-1:2

  if(x.is_leaf(ii))
    
    % Set U{ii} to Q
    x.U{ii} = Q_cell{ii};
    
  else
    % apply R{ii_left}, R{ii_right}:  
    ii_left  = x.children(ii, 1);
    ii_right = x.children(ii, 2);
    
    R_left_cell  = mat2cell(R_cell{ii_left }, ...
			    size(R_cell{ii_left }, 1), ...
			    blk_rank(ii_left , :));
    R_right_cell = mat2cell(R_cell{ii_right}, ...
			    size(R_cell{ii_right}, 1), ...
			    blk_rank(ii_right, :));
    
    x.B{ii} = cell(1, 1, s);
    for jj=1:s
      x.B{ii}{jj} = ttm(x_cell{jj}.B{ii}, ...
			{R_left_cell{jj}, ...
			 R_right_cell{jj}}, [1 2]);
    end
    
    x.B{ii} = cell2mat(x.B{ii});
    
    % Matricize B{ii}
    B_mat = matricize(x.B{ii}, [1 2], 3);
    
    % Calculate QR-decomposition
    [Q, R] = qr(B_mat, 0);
    
    R_cell{ii} = R;
    
    % Calculate dimensions of "tensor" Q
    tsize_new = size(x.B{ii});
    tsize_new(3) = size(Q, 2);
    
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


% Apply R_cell to root node B
ii_left  = x.children(1, 1);
ii_right = x.children(1, 2);

R_left_cell  = mat2cell(R_cell{ii_left }, ...
			size(R_cell{ii_left }, 1), blk_rank(ii_left , :));
R_right_cell = mat2cell(R_cell{ii_right}, ...
			size(R_cell{ii_right}, 1), blk_rank(ii_right, :));

x.B{1} = ttm(x_cell{1}.B{1}, {R_left_cell{1}, R_right_cell{1}}, [1 2]);
for jj=2:s
  x.B{1} = x.B{1} + ttm(x_cell{jj}.B{1}, {R_left_cell{jj}, R_right_cell{jj}}, [1 2]);
end

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
  err_ = err; err_(ht.children(1, 1)) = 0;
  
  % Calculate upper bound and factor c from ||x - x_|| <= c ||x - x_best||
  err_bd = norm(err_); factor = sqrt(2*ndims(ht)-3);
  
  fprintf(['\nLower/Upper bound for best approximation error:\n' ...
	   '%e <= ||X - X_best|| <= ||X - X_|| <= %e\n'], ...
	  max(err_bd/factor, max(err)), err_bd);
end
