function [ht, err, sv] = truncate(x, opts)
%TRUNCATE Truncates full tensor/htensor/CP-decomposition to an htensor.
%
%   Y = TRUNCATE(X, OPTS) truncates tensor X to lower-rank htensor
%   Y, depending on the options:
%
%   1) No rank k(ii) can be bigger than OPTS.MAX_RANK. 
%   2) The relative error in tensor norm cannot be bigger than
%   OPTS.REL_EPS, except when the first condition requires it.
%   3) The absolute error in tensor norm cannot be bigger than
%   OPTS.ABS_EPS, except when the first condition requires it.
%
%   Only OPTS.MAX_RANK is required, the other conditions are
%   optional.
%
%   This is a wrapper function that calls the appropriate truncation
%   depending on the class of x: 
%   - MATLAB array (i.e. full tensor): truncate_ltr
%   - htensor (tensor in HTD decomposition): truncate_std
%   - ktensor (CP decomposition, Tensor Toolbox): truncate_cp
%   - cell array (CP decomposition): truncate_cp
%
%   The expected errors in each node and overall are displayed if
%   OPTS.PLOT_ERRTREE is set to true.
%
%   See also HTENSOR, ORTHOG, GRAMIANS

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

if(nargin ~= 2)
  error('Requires exactly 2 arguments.')
end

if(~isa(opts, 'struct') || ~isfield(opts, 'max_rank') )
  error(['Second argument must be a MATLAB struct with field max_rank,' ...
	 ' and optionally abs_eps and/or rel_eps.']);
end

if(isnumeric(x))
  [ht, err, sv] = htensor.truncate_ltr(x, opts);
elseif(isa(x, 'htensor'))
  [ht, err, sv] = truncate_std(x, opts);
elseif(isa(x, 'ktensor'))
  [ht, err, sv] = htensor.truncate_cp(x, opts);
elseif(isa(x, 'cell'))
  [ht, err, sv] = htensor.truncate_cp(x, opts);
else
  error('No truncation function available for %s.', class(x));
end
