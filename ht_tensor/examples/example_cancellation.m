function example_cancellation()
%EXAMPLE_CANCELLATION Cancellation in tan(x) + 1/x - tan(x).
%
% Calculate exp(-x^2) + sin(x)^2 + cos(x)^2, using successive
% truncation and directly.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

disp(['Calculate tan(x) + 1/x - tan(x), using successive truncation' ...
      ' and directly.']);

d = 3;
n = 100;
opts.max_rank = 20;

t = linspace(0, 1, n+1)';
t = t(2:end);

% Construct x1+x2+x3
for ii=1:d
  sum_t{ii} = ones(n, d);
  sum_t{ii}(:, ii) = t;
end
sum_full = full(htensor(sum_t));

% Construct 1/(x1+x2+x3)
inv_sum_full = 1./sum_full;
inv_sum_htd = truncate(inv_sum_full, opts);

% Construct tan(x1+x2+x3)
tan_full = tan(sum_full);
tan_htd = truncate(tan_full, opts);


sum_exact = tan_htd + inv_sum_htd - tan_htd;

sum_trunc_standard = truncate(tan_htd + inv_sum_htd - tan_htd, ...
			      opts);

sum_trunc_sum = htensor.truncate_sum({tan_htd, inv_sum_htd, -tan_htd}, ...
			      opts);

sum_succ_trunc = truncate(tan_htd + inv_sum_htd, opts);
sum_succ_trunc = truncate(sum_succ_trunc - tan_htd, opts);

err_std_trunc  = norm(sum_exact - sum_trunc_standard)/norm(sum_exact)
err_trunc_sum  = norm(sum_exact - sum_trunc_sum)/norm(sum_exact)
err_succ_trunc = norm(sum_exact - sum_succ_trunc)/norm(sum_exact)

disp(['Note that this is a constructed example; in many cases,' ...
      ' successive truncation results in similar errors as' ...
       ' all-at-once truncation.']);