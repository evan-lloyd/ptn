function [k, err, success] = trunc_rank(s, opts)
%TRUNC_RANK Rank to truncate at to fulfill requirements.
%
%   TRUNC_RANK Returns the rank K to truncate singular values S at,
%   given the requirements in OPTS:
%
%   1) The rank K cannot be bigger than OPTS.MAX_RANK. 
%   2) The relative error in tensor norm cannot be bigger than
%   OPTS.REL_EPS, except when the first condition requires it.
%   3) The absolute error in tensor norm cannot be bigger than
%   OPTS.ABS_EPS, except when the first condition requires it.
%
%   Only OPTS.MAX_RANK is a required argument (though it is ignored
%   when set to Inf). The other too arguments are optional.
%
%   When OPTS.PLOT_SV is set to TRUE, a plot with all conditions is
%   generated.
%
%   The output ERR gives the error in tensor norm, and boolean
%   SUCCESS indicates if all requirements are met.
%
%   See also: HTENSOR, TRUNCATE.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

if(~isnumeric(s) || ~isvector(s))
  error('First argument must be a vector.');
end

if(~isa(opts, 'struct') || ~isfield(opts, 'max_rank') )
  error(['Second argument must be a MATLAB struct with field max_rank,' ...
	 ' and optionally fields abs_eps and/or rel_eps.']);
end

% When truncating at k, error in tensor norm is s_sum(k+1):
% s_sum(k+1) = norm(s(k+1:end))
s_sum = sqrt(cumsum(s(end:-1:1).^2));
s_sum = s_sum(end:-1:1);

% Calculate necessary rank to satisfy relative eps
if(~isfield(opts, 'rel_eps'))
  k_rel = [];
else
  k_rel = find(s_sum < opts.rel_eps*norm(s), 1, 'first');
  k_rel = k_rel-1;
  if(numel(k_rel) == 0)
    k_rel = length(s);
  end
end

% Calculate necessary rank to satisfy absolute eps
if(~isfield(opts, 'abs_eps'))
  k_abs = [];
else
  k_abs = find(s_sum < opts.abs_eps, 1, 'first');
  k_abs = k_abs-1;
  if(numel(k_abs) == 0)
    k_abs = length(s);
  end
end

opts.max_rank = min([opts.max_rank, numel(s)]);

% Calculate necessary rank to satisfy absolute and relative eps
k_eps = max([k_rel, k_abs]);

% Use k_eps if <= max. rank, otherwise use max. rank
k = min([k_eps, opts.max_rank]);
k = max(k, 1);

% Read error from s_sum
s_sum = [s_sum; 0];
err = s_sum(k+1);

% Warning message when relative or absolute eps is not preserved.
if(k < k_eps)
  success = false;
else
  success = true;
end

% For debugging / illustration: Plot the values s, rel_eps, abs_eps
% and the truncation rank k.
if(isfield(opts, 'plot_sv') && opts.plot_sv)
  semilogy(0:length(s_sum)-1, s_sum, 'b-x');
  xlim([0 length(s)])
  hold on;
  semilogy([0 length(s)], opts.rel_eps*norm(s)*[1 1], 'r');
  semilogy([0 length(s)], opts.abs_eps*[1 1], 'g');
  if(opts.max_rank <= length(s))
    semilogy(opts.max_rank*[1 1], ylim(), 'k');
  end
  semilogy(k, s_sum(k+1), 'ro');
  
  legend('error decay', 'rel.eps', 'abs.eps', 'max.rank') 
  drawnow;
end
