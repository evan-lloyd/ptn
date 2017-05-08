function example_elem_reciprocal()
%EXAMPLE_ELEM_RECIPROCAL Calculation of element-wise reciprocal of htensor.
%
%   See also HTENSOR.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

disp('c = laplace_core(4);');
c = laplace_core(4);

disp('U = [ones(100, 1), linspace(1e-3, 1, 100)]''];');
U = [ones(100, 1), linspace(1e-3, 1, 100)'];

disp('x = ttm(c, {U, U, U, U});');
x = ttm(c, {U, U, U, U});

opts.max_rank = 7;
opts.abs_eps = 1e-6;
opts.rel_eps = 1e-6;

opts.elem_mult_max_rank = 30;
opts.elem_mult_abs_eps = 1e-16;

opts.maxit = 30;
opts.verbose = true;

opts

keyboard;

figure;
disp('inv_x = elem_reciprocal(x, opts);');
disp('-----------------------------------');
inv_x = elem_reciprocal(x, opts);
disp('-----------------------------------');
title('Convergence without knowing maximal entry');

keyboard;

disp('Upper bound for maximal element of x:');
max_x = 4*max(U(:, 2));

figure;
disp('inv_x = elem_reciprocal(x, opts, max_x);');
disp('-----------------------------------');
inv_x = elem_reciprocal(x, opts, max_x);
disp('-----------------------------------');
title('Convergence with maximal entry');

disp('plot_sv(inv_x);');
plot_sv(inv_x);
