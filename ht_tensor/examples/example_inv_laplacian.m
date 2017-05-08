function example_inv_laplacian()
%EXAMPLE_INV_LAPLACIAN Generating approximate inverse Laplacian.
%
%   See also GEN_INV_LAPLACIAN.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

disp('The d-dimensional Laplacian has the Kronecker structure:');
disp('A = Lapl x I x ... x I + ... + I x ... x I x Lapl');
disp('Eigenvalue decomposition leads to:');
disp(['inv(A) = (iS x ... x iS) inv(L x I x ... x I + ... + I x ...' ...
      ' x I x L) (S x ... x S)']);
disp(['L are diagonal matrices with known entries, and S, iS the' ...
      ' discrete Sine transformation and its inverse.']);
disp(['Given an htensor approximating the diagonal entries of this' ...
      ' inverse matrix, the inverse laplacian can be efficiently' ...
      ' applied to an htensor:']);
disp('x_ = ttm(invLambda.*ttm(b, {@dst, @dst, @dst}), {@idst, @idst, @idst})');
disp(' ');

disp('n = 100; d = 7;'); disp(' ');
n = 100; d = 7;

% Eigenvalues of 1D-Laplacian spdiags([-ones(n, 1), 2*ones(n, 1),
% -ones(n, 1)], [-1 0 1], n, n)*(n+1)^2;
disp('Construct htensor containing all eigenvalues of the Laplacian:')

disp('lambda1d = 4*sin(pi*(1:n)''/(2*(n+1))).^2*(n+1)^2;');
lambda1d = 4*sin(pi*(1:n)'/(2*(n+1))).^2*(n+1)^2;

disp('Lambda = laplace_core(d);');
disp('for ii=1:d');
disp('  Lambda = ttm(Lambda, [ones(size(lambda1d)), lambda1d], ii);');
disp('end'); disp(' ');

Lambda = laplace_core(d);
for ii=1:d
  Lambda = ttm(Lambda, [ones(size(lambda1d)), lambda1d], ii);
end

disp('Tensor containing only ones:');
disp('all_ones = cell(1, d);');
disp('for ii=1:d');
disp('  all_ones{ii} = ones(size(lambda1d));');
disp('end');
disp('all_ones = htensor(all_ones);');
disp(' ');
all_ones = cell(1, d);
for ii=1:d
  all_ones{ii} = ones(size(lambda1d));
end
all_ones = htensor(all_ones);

disp('M = 120; opts.max_rank = 15;');
M = 120; opts.max_rank = 15;

disp('invLambda = gen_inv_laplacian(n*ones(1, d), M, opts);');
invLambda = gen_inv_laplacian(n*ones(1, d), M, opts);

fprintf('\n|| invLambda .* Lambda - all_ones || / || all_ones || = %e\n', ...
norm(orthog(invLambda.*Lambda - all_ones))/norm(all_ones));

disp(['The approximation error of invLambda is considered in' ...
      ' approx_inv_laplacian.m']);
