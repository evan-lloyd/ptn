function x =  elem_reciprocal(y, opts, max_y)
%ELEM_RECIPROCAL Iteratively find elementwise reciprocal of Y.
%
%  X = ELEM_RECIPROCAL(Y, OPTS, [MAX_Y]) calculates X = 1./Y
%  iteratively, using the Newton-Schulz iteration:
%    X_(k+1) = X_k + X_k.*(1 - Y.*X_k)
%
%  As the element-wise product results in an htensor of squared
%  ranks, truncating is important for this algorithm. There is a
%  risk of leaving the convergence space of the Newton-Schulz
%  iteration by truncations, mainly in the beginning.
%
%  The starting htensor is chosen as X_0 = 2/norm(Y)*ALL_ONES, though
%  a better approximation is X_0 = 1/max(Y)*ALL_ONES, which may be
%  supplied by the user. As the convergence is quadratic near 1./Y, a
%  good starting value is very important. Convergence is increased
%  for any upper bound MAX_Y fulfilling
%   max(Y) <= MAX_Y < norm(Y)/2.
%
%  Required fields of OPTS:
%  - ELEM_MULT_MAX_RANK      Option for ELEM_MULT
%  - ELEM_MULT_ABS_EPS       Option for ELEM_MULT
%  - MAX_RANK                Option for TRUNCATE
%  - MAXIT                   Maximal number of iterations

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

opts_elem_mult.max_rank = opts.elem_mult_max_rank;
opts_elem_mult.abs_eps = opts.elem_mult_abs_eps;

if(~isfield(opts, 'verbose'))
  opts.verbose = false;
end

for ii=1:ndims(y)
  all_ones{ii} = ones(size(y, ii), 1);
end
all_ones = htensor(all_ones);

% Best starting value would be 2/(max_y + min_y), but any smaller
% value leads to convergence, ignoring the truncation errors.
if(nargin == 3)
  x = 2/(2*max_y)*all_ones;
else
  x = 2/norm(y)*all_ones;
end

for ii=1:opts.maxit
  
  [xy, err_delta_elmul] = elem_mult(x, y, opts_elem_mult);
  
  [delta, err_delta_trunc] = truncate(all_ones - xy, opts); 
  
  [deltax, err_x_elmul] = elem_mult(delta, x, opts_elem_mult);
  
  [x, err_x_trunc] = truncate(x + deltax, opts);
  
  if(opts.verbose)
    rel_delta(ii) = norm(delta)/norm(all_ones)
    rank_x = rank(x)
    rank_delta = rank(delta)
    
    semilogy(rel_delta);
    ylabel('||y*xk - 1||/||1||')
    drawnow;
  end
  
  if(ii>1 && rel_delta(ii) > rel_delta(ii-1))
    break;
  end
  
end
