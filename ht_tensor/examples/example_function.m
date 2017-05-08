function example_function()
%EXAMPLE_FUNCTION Demonstration of htensor function approximation.
%
%   See also HTENSOR.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

disp('Construct full order-5 tensor x = 1/(t1 + ... + t5):');
n = 15;
tic;
x = zeros(n, n, n, n, n);
t = linspace(0, 1, n+1);
t = t(2:end)

shp = ones(1, ndims(x));
shp(1) = n;

rep = n*ones(1, ndims(x));
rep(1) = 1;

for ii=1:ndims(x)
  x = x + repmat(reshape(t, shp), rep);
  shp = [shp(end), shp(1:end-1)];
  rep = [rep(end), rep(1:end-1)];
end
x = 1./x;
disp('> size(x)')
size(x)
fprintf('Elapsed time: %f s\n', toc);
keyboard;

norm_x = norm_nd(x);

disp('> opts.max_rank = 100;');
disp('> opts.abs_eps = 1e-5;');
disp('> opts.plot_errtree = true;');
opts.max_rank = 100;
opts.abs_eps = 1e-5;
opts.plot_errtree = true;

disp('> x_trunc = truncate(x, opts);')
x_trunc = truncate(x, opts);

disp('> norm_nd(full(x_trunc) - x)/norm_nd(x)');
norm_nd(full(x_trunc) - x)
keyboard;

disp('Plot effective relative error against rel_eps:')
h = 10.^[-2:-2:-12];

for ii=1:length(h)
  
  opts2.max_rank = 30;
  opts2.rel_eps = h(ii);
  
  tic;
  x_ltr = truncate(x, opts2);
  time_ltr(ii) = toc;
  
  s = whos('x_ltr');
  size_ltr(ii) = s.bytes;
  
  err_norm_ltr(ii) = norm_diff(x_ltr, x)/norm_x;
  
  rel_eps = opts2.rel_eps
  rank_truncated = rank(x_ltr)
  
  subplot(3, 1, 1);
  loglog(h(1:ii), err_norm_ltr, 'bx-', h(1:ii), h(1:ii), 'r');
  xlabel('rel. eps');
  ylabel('relative error')
  
  subplot(3, 1, 2);
  semilogx(h(1:ii), size_ltr(1:ii)/1000, 'bx-');
  xlabel('rel. eps');
  ylabel('Storage (KB)');

  subplot(3, 1, 3);
  semilogx(h(1:ii), time_ltr, 'bx-');
  xlabel('rel. eps');
  ylabel('execution time');
  
  drawnow;
  
end

keyboard;

disp('> opts3.max_rank = 3;');
disp('> x3 = truncate(x, opts3);');
opts3.max_rank = 3;
x3 = truncate(x, opts3);

disp('> opts3.max_rank = 2;');
disp('> x2 = truncate(x, opts3);');
opts3.max_rank = 2;
x2 = truncate(x, opts3);

disp('> x2trunc3 = truncate(orthog(x3), opts3);');
x2_ = truncate(orthog(x3), opts3);

fprintf('||x3 - x2||/||x3||: %e\n', norm(orthog(x3 - x2))/ ...
	norm(x3));

fprintf('||x3 - x2_||/||x3||: %e\n', norm(orthog(x3 - x2_))/ ...
	norm(x3));
