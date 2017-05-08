function [x, norm_r] = cg_tensor(apply_mat, apply_precond, b, opts)
%CG_TENSOR Truncated conjugate gradients method for htensors.
%
%   [X, NORM_R] = CG_TENSOR(APPLY_MAT, APPLY_PRECOND, B, OPTS)
%   applies the CG method to the system 
%      APPLY_MAT(X) = B,
%   using the preconditioner APPLY_PRECOND. The iterands are
%   truncated in each iteration.
%
%   Required fields of OPTS:
%   - OPTS.MAXIT       maximum number of iterations
%   - OPTS.TOL         tolerance for result
%   - OPTS.REL_TOL     stop when error increases by a certain factor
%   - OPTS.PLOT_CONV   whether to plot residual during iteration
%   - OPTS.MAX_RANK    argument to TRUNCATE
%
%   Optional fields of OPTS:
%   - OPTS.ABS_EPS     argument to TRUNCATE
%   - OPTS.REL_EPS     argument to TRUNCATE

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

n = size(b);
d = ndims(b);

x = 0*b;

r = b;
z = apply_precond(r, opts);
p = z;


Ap = apply_mat(p, opts);

%rz = innerprod(r, z);
pAp = innerprod(p, Ap);

norm_r(1) = norm(r)/norm(b)

err_p = 0;

for ii = 2:opts.maxit
    ii
  omega = innerprod(r, p)/pAp;
  %omega = rz/innerprod(p, Ap);
  
  x = x + p*omega;
  [x, err_x] = truncate(x, opts);
    
  %r = r - Ap*omega;
  %[r, err_r] = truncate(r, opts);
    
  if(opts.max_rank > 1)
    r = b - apply_mat(x, opts);
  else
    r = b - apply_mat(x);
  end
  r = orthog(r);
  
  norm_r(ii) = norm(r)/norm(b);
  %norm_r_real(ii) = norm(b - apply_mat(x, opts))/norm(b);
  
  r = truncate(r, opts);
    
  if(opts.plot_conv)
    semilogy(norm_r, 'b');
    hold on;
    drawnow;
  end  
  
  if( norm_r(ii) < opts.tol ...
      || norm_r(ii-1) - norm_r(ii) < opts.rel_tol*norm_r(ii) )
    break;
  end
  z = apply_precond(r, opts);
  
  %rz_new = innerprod(r, z);
  %beta = rz_new/rz;
  %rz = rz_new;
  beta = -innerprod(z, Ap)/pAp;
  
  p = z + p*beta;
  [p, err_p] = truncate(p, opts);
    
  Ap = apply_mat(p, opts);
  pAp = innerprod(p, Ap);
end
