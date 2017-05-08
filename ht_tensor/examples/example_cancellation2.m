function example_cancellation2()
%EXAMPLE_CANCELLATION2 Cancellation in exp(-x^2) + sin(x)^2 + cos(x)^2.
%
% Calculate exp(-x^2) + sin(x)^2 + cos(x)^2, using successive
% truncation and directly.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

disp(['Calculate exp(-x^2) + sin(x)^2 + cos(x)^2, using successive' ...
      ' truncation' ' and directly.']);

d = 3;
n = 100;

t = linspace(0, 1, n+1)';
t = t(2:end);

% Construct x1+x2+x3
for ii=1:d
  sum_dir{ii} = ones(n, d);
  sum_dir{ii}(:, ii) = t;
  sum_sqr{ii} = ones(n, d);
  sum_sqr{ii}(:, ii) = t.^2;
end
sum_dir_full = full(htensor(sum_dir));
sum_sqr_full = full(htensor(sum_sqr));

% Construct sin(x1+x2+x3)^2
gaussian_full = exp(-sum_sqr_full);

opts.max_rank = 4;
gaussian_htd = htensor.truncate_ltr(gaussian_full, opts);

sin2_full = sin(sum_dir_full).^2;
sin2_htd = htensor.truncate_ltr(sin2_full, opts);

% Construct cos(x1+x2+x3)^2
cos2_full = cos(sum_dir_full).^2;
cos2_htd = htensor.truncate_ltr(cos2_full, opts);

opts.max_rank = 2;

sum_exact = gaussian_htd + sin2_htd + cos2_htd;

sum_trunc_standard = truncate(sum_exact, opts);

sum_trunc_sum = htensor.truncate_sum(...
    {gaussian_htd, sin2_htd, cos2_htd}, opts);

sum_succ_trunc = truncate(gaussian_htd + sin2_htd, opts);
sum_succ_trunc = truncate(sum_succ_trunc + cos2_htd, opts);

err_std_trunc  = norm(sum_exact - sum_trunc_standard)/norm(sum_exact)
err_trunc_sum  = norm(sum_exact - sum_trunc_sum)/norm(sum_exact)
err_succ_trunc = norm(sum_exact - sum_succ_trunc)/norm(sum_exact)

disp(['Note that this is a constructed example; in many cases,' ...
      ' successive truncation results in similar errors as' ...
       ' all-at-once truncation.']);
