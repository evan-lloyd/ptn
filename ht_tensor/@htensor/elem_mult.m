function [z, err, sv] = elem_mult(x, y, opts)
%ELEM_MULT Element-by-element multiplication of htensors, with truncation.
%
%   Z = ELEM_MULT(X, Y, OPTS) Elementwise multiplication of htensors X
%   and Y. Both must have identical dimension trees and the same
%   size. The result is truncated during construction, according to
%   OPTS. Note that this truncation is not optimal, and that only
%   the fields OPTS.MAX_RANK and OPTS.ABS_EPS are used.
%
%   The routine Z = TIMES(X, Y) returns the exact result, but that
%   operation is much more expensive.
%
%   See also HTENSOR/TIMES.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

% If opts is not given, calculate exactly
if(nargin == 2)
  z = x.*y;
  return;
end

% Check number of arguments
if(nargin ~= 3)
  error('Requires 2 or 3 arguments.');
end

% Check compatibility of dimension trees.
if(~equal_dimtree(x, y))
  error('Dimension trees of X and Y differ.');
end

% Check sizes
if(~isequal(size(x), size(y)))
  error('Tensor dimensions must agree.')
end

% Check opts
if(~isa(opts, 'struct') || ~isfield(opts, 'max_rank') )
  error(['Third argument must be a MATLAB struct with field max_rank,' ...
	 ' and optionally fields abs_eps and/or rel_eps.']);
end

% opts.rel_eps has no influence:
if(isfield(opts, 'rel_eps'))
  opts = rmfield(opts, 'rel_eps');
end

% Orthogonalize x and y if necessary
x = orthog(x);
y = orthog(y);

% Calculate Gramians
Gx = gramians(x);
Gy = gramians(y);

% Initialize result
z = x;

% err represents the node-wise truncation errors
err = zeros(z.nr_nodes, 1);

for ii=2:z.nr_nodes
  
  % Calculate the left singular vectors U_ and singular values s
  % of X_{ii} from the gramian G{ii}.
  [U_x{ii}, sv_x] = htensor.left_svd_gramian(Gx{ii});
  [U_y{ii}, sv_y] = htensor.left_svd_gramian(Gy{ii});
  
  % Calculate all singular values:
  SV = sv_x*sv_y';
  [sv{ii}, subs] = sort(SV(:), 'descend');
  
  % Calculate rank k to use, and expected error.
  [k, err(ii)] = htensor.trunc_rank(sv{ii}, opts);
  [ind_x, ind_y] = ind2sub(size(SV), subs(1:k));
  
  % Truncate U_x, U_y; S_ = U_x(:, ii) kron U_y(:, ii), ii=1:k
  U_x{ii} = U_x{ii}(:, ind_x);
  U_y{ii} = U_y{ii}(:, ind_y);
  
end

for ii=z.nr_nodes:-1:2

  % Apply U_ to node ii:
  if(z.is_leaf(ii))
    Ux = x.U{ii}*U_x{ii};
    Uy = y.U{ii}*U_y{ii};
    z.U{ii} = Ux.*Uy;
  else
    ii_left  = z.children(ii, 1);
    ii_right = z.children(ii, 2);
    
    Bx = ttm(x.B{ii}, {U_x{ii_left}, U_x{ii_right}}, [1 2], 'h');
    Bx = ttm(Bx, U_x{ii}, 3, 't');
    
    By = ttm(y.B{ii}, {U_y{ii_left}, U_y{ii_right}}, [1 2], 'h');
    By = ttm(By, U_y{ii}, 3, 't');
        
    z.B{ii} = Bx.*By;
  end
  
end

ii = 1;
ii_left  = z.children(ii, 1);
ii_right = z.children(ii, 2);

Bx = ttm(x.B{ii}, {U_x{ii_left}, U_x{ii_right}}, [1 2], 'h');
By = ttm(y.B{ii}, {U_y{ii_left}, U_y{ii_right}}, [1 2], 'h');

z.B{ii} = Bx.*By;

z.is_orthog = false;
