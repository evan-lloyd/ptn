function approx_inv_laplacian(n)
%APPROX_INV_LAPLACIAN Error-rank dependence of inverse Laplacian.
%
% Plots the rank-dependence of the eigenvalues of the Laplacian,
% where (2M+1) coefficients are used in the quadrature rule, and
% the result is truncated back to a smaller rank.
%
% The error is given as the relative residual
% || diag(Lambda) * invLambda - all_ones|| ...
%        / (max(Lambda)*||invLambda|| + ||all_ones||).
%
% This is not ideal, as we are actually interested in 
% max( Lambda.^(-1) - invLambda )
% which is not readily available.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

if(~exist('n', 'var'))
  n = 100;
end

file_name = sprintf('approx_inv_laplacian_n%d.mat', n);

if(~exist(file_name, 'file'))
  gen_data(n, file_name);
else
  usr_str = input(['Data from previous run available, recalculate' ...
		   ' (y/N)? '], 's');
  
  if(numel(usr_str) == 0)
    usr_str = 'n';
  end
  
  if(usr_str(1) == 'y' || usr_str(1) == 'Y')
    gen_data(n, file_name);
  end

end

load(file_name);
clf;

for ind_d=1:length(d_values)
  subplot(2, 1, 1);
  semilogy(rk_values, rel_res(ind_d, :) , [col(ind_d) 'x-']);
  hold on;
  subplot(2, 1, 2);
  plot(rk_values, M_max(ind_d, :) , [col(ind_d) 'x-']);
  hold on;
end

subplot(2, 1, 1);
legend('ndims 3', 'ndims 6', 'ndims 10', 'ndims 20');
xlabel('max. rank')
ylabel('rel.error')

subplot(2, 1, 2);
legend('ndims 3', 'ndims 6', 'ndims 10', 'ndims 20');
xlabel('max. rank')
ylabel('M (quadrature rule)')


function gen_data(n, file_name)

d_values = [3, 6, 10, 20];
M_start = 20;
rk_values = 3:2:20;
col = 'brgkcm';

% Initialize Laplacian
% Eigenvalues of spdiags([-ones(n, 1), 2*ones(n, 1), -ones(n, 1)], ...
% [-1 0 1], n, n)*(n+1)^2;
lambda1d = 4*sin(pi*(1:n)'/(2*(n+1))).^2*(n+1)^2;

rel_res = zeros(numel(d_values), numel(rk_values));
M_max = NaN(numel(d_values), numel(rk_values));
for ind_d=1:length(d_values)
  d = d_values(ind_d);
  
  Lambda = laplace_core(d);
  all_ones = cell(1, d);
  for ii=1:d
    Lambda = ttm(Lambda, [ones(size(lambda1d)), lambda1d], ii);
    all_ones{ii} = ones(size(lambda1d));
  end
  all_ones = htensor(all_ones);
  
  M = M_start;
  
  for ind_rk=1:length(rk_values)
	
    opts.max_rank = rk_values(ind_rk);
    
    iter = 1;
    rel_res_ii = [];
    while( iter < 20 && ...
	   (iter < 3 || rel_res_ii(end)/rel_res_ii(end-1) < 0.9))
      
      invLambda = gen_inv_laplacian(n*ones(1, d), M, opts);
      rel_res_ii(iter) = norm(Lambda.*invLambda - all_ones)/ ...
	  (d*max(lambda1d)*norm(invLambda) + norm(all_ones));
      iter = iter + 1;
      M = M + M_start;
    end
    rel_res(ind_d, ind_rk) = rel_res_ii(end);
    M_max(ind_d, ind_rk) = M;
    M = max(M - 2*M_start, M_start);
    
    if(ind_rk > 1 && rel_res(ind_d, ind_rk-1) < rel_res(ind_d, ...
							ind_rk))
      break;
    end
    
    subplot(2, 1, 1);
    semilogy(rk_values(1:ind_rk), rel_res(ind_d, 1:ind_rk), ...
	     [col(ind_d) 'x-']);
    hold on;
    subplot(2, 1, 2);
    plot(rk_values(1:ind_rk), M_max(ind_d, 1:ind_rk), ...
	 [col(ind_d) 'x-']);
    hold on;
    drawnow;
  end
end

save(file_name, 'd_values', 'rk_values', 'M_max', 'rel_res', 'col');