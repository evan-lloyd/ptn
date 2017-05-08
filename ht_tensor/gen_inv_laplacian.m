function [C, U] = gen_inv_laplacian(A, k, opts)
%GEN_INV_LAPLACIAN Approximate inverse of Laplacian matrix.
%
%  [C, U] = GEN_INV_LAPLACIAN(A, k, OPTS) calculates an
%  approximation of the inverse of matrix Lapl:
%
%  Lapl = A{1} x I x ... x I + ... + I x ... x I x A{d}
%
%  Cell array U contains the eigenvectors of Lapl in each dimension,
%  while C is an approximation of the inverse of Lapl's
%  eigenvalues. This approximation is done by sinc-quadrature,
%  using (2k+1) factors, and the resulting tensor C is truncated
%  according to OPTS.
%
%  The approximate inverse can be applied to a tensor x as follows:
%
%  Laplinv_x = ttm(full(C).*ttm(x, U, 'h'), U); % hermitian A{i}
%
%  for ii=1:d, inv_U{ii} = @(v)(U{ii}\v); end;  % diagonalizable A{i}
%  Laplinv_x = ttm(full(C).*ttm(x, inv_U), U);
%  
%  C = GEN_INV_LAPLACIAN(N, k, OPTS), where N is a vector of
%  positive integers, is the special case where A{ii} are the
%  Laplacian matrices of size N(ii):
%
%  A = (n+1)^2*spdiags(ones(n, 1)*[-1, 2, -1], [-1 0 1], n, n);
%
%  This is a special case, as U corresponds to the Discrete Sine
%  Transformation, and the values of lambda are known
%  analytically.
%
% Examples:
%
% x = randn(5, 5, 5); opts.max_rank = 15;
% Lapl = spdiags([-ones(5, 1), 2*ones(5, 1), -ones(5, 1)], ...
% 	      [-1 0 1], 5, 5)*(5+1)^2;
% C = gen_inv_laplacian([5 5 5], 10, opts);
% Laplx = ttm(x, A, 1) + ttm(x, A, 2) + ttm(x, A, 3);
% x_ = ttm(full(C).*ttm(Laplx, {@dst, @dst, @dst}), ...
%	   {@idst, @idst, @idst});
%
% A = randn(5); A = A*A';
% [C, U] = gen_inv_laplacian({A, A, A}, 10, opts);
% Laplx = ttm(x, A, 1) + ttm(x, A, 2) + ttm(x, A, 3);
% x_ = ttm(full(C).*ttm(Laplx, U, 'h'), U);

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

if(isa(A, 'cell'))

  d = numel(A);
  n = cellfun('size', A, 1);
  if(cellfun('size', A, 2) ~= n)
    error('A must contain quadratic matrices.')
  end

  % Diagonalize A{ii}
  for ii=1:d
    [U{ii}, D] = eig(A{ii});
    lambda{ii} = diag(D);
  end
  
elseif(isindexvector(A))
  n = A;
  d = numel(n);
    
  % Initialize eigenvalues of FDM 1d-Laplacian matrix
  for ii=1:d
    lambda{ii} = 4*sin(pi*(1:n(ii))'/(2*(n(ii)+1))).^2*(n(ii)+1)^2;    
  end
end

% Quadrature rule:
k_vec = (-k:k);
hst = pi/sqrt(k);
t = log(exp(k_vec*hst)+sqrt(1+exp(2*k_vec*hst)));
w = hst./sqrt(1+exp(-2*k_vec*hst));

lambdaMin = sum(cellfun(@min, lambda));
alpha = -2*t/lambdaMin;
omega = 2*w/lambdaMin;

% CP decomposition:
cell_cols = cell(1, d);
for ii=1:d
  cell_cols{ii} = exp(lambda{ii}*alpha);
end

% Truncate from CP decomposition to htensor of smaller rank
C = htensor.truncate_cp(cell_cols, opts, omega);
