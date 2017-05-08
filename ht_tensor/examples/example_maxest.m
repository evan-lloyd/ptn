function example_maxest()
%EXAMPLE_MAXEST Calculation of maximum element of htensor.
%
%   See also HTENSOR.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

disp('x = htenrandn([40 40 40]);')
x = htenrandn([40 40 40]);

opts.max_rank = 7;
opts.abs_eps = 1e-6;
opts.rel_eps = 1e-6;

opts.elem_mult_max_rank = 30;
opts.elem_mult_abs_eps = 1e-16;

opts.maxit = 30;
opts.verbose = true;

opts

keyboard;

disp('[max_x, sub] = maxest(x, opts)');
disp('-----------------------------------');
[max_x, sub] = maxest(x, opts);
disp('-----------------------------------');

max_x
sub

[max_exact, ind] = max(abs(x(:)));
[sub1, sub2, sub3] = ind2sub(size(x), ind);

max_exact = x(sub1, sub2, sub3)
sub_exact = [sub1, sub2, sub3]