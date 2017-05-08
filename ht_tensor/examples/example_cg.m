function [A, b_tensor, x, alpha, M, Mesh, U_bd, norm_r_cg] = example_cg
%EXAMPLE_CG Apply CG method to a parametric PDE.
%
%   See also HTENSOR.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

disp('THIS WILL TAKE SOME TIME:')

% Load the system matrices
load cookies_matrices_2x2.mat
%load cookies_matrices_3x3.mat

% Plot Mesh
figure;
patch('faces', Mesh.Elements, ...
      'vertices', Mesh.Coordinates, ...
      'facevertexcdata', Mesh.ElemFlag, ...
      'facecolor', 'flat', ...
      'edgecolor', 'black');

axis tight; axis equal;

% System size
n = length(A{1});

% Number of parameters
p = length(A) - 1;

% Initialize parameter samples alpha in each direction
alpha{1} = [];
for jj=2:p+1
alpha{jj} = 0:100;
m(jj) = length(alpha{jj});
end

% Initialize system matrix handle
A_handle = handle_lin_mat(A, alpha);

% Initialize right-hand side b
b_cell = cell(1, p+1);
b_cell{1} = b;
for jj=2:p+1
  b_cell{jj} = ones(m(jj), 1);
end
b_tensor = htensor(b_cell);

% Initialize Preconditioner
M = A{1};
for ii=2:p+1
M = M + mean(alpha{ii})*A{ii};
end
M_handle = handle_inv_mat(M);

% Initialize opts structure
rk_vals = [10, 20, 30]; col = 'brgcm';
opts.rel_eps = 1e-10;

opts.maxit = 50;
opts.tol = 0;
opts.rel_tol = -Inf;

opts.disp = 0;
opts.plot_errtree = false;
opts.plot_conv = true;
opts.tree_balance = 'standard';

figure;

% Start conjugate gradient method
for ii=1:numel(rk_vals)
  
  opts.max_rank = rk_vals(ii);
  
  fprintf('MAX. RANK %d', opts.max_rank);
  tic;
  [X{ii}, norm_r{ii}] = cg_tensor(A_handle, M_handle, b_tensor, opts);
  toc
  
  semilogy(norm_r{ii}, col(ii));
  
end

clf;

semilogy(1:opts.maxit, norm_r{1}, col(1), ...
	 1:opts.maxit, norm_r{2}, col(2), ...
	 1:opts.maxit, norm_r{3}, col(3));

legend('rank 10', 'rank 20', 'rank 30');
xlabel('iterations');
ylabel('||Ax - b||/||b||');
