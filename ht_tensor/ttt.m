function [z, ind_compl_x, ind_compl_y] = ttt(x, y, ind_x, ind_y)
%TTT Tensor times tensor (full tensors).
%
%  TTT(X,Y) computes the outer product of tensors X and Y.
% 
%  TTT(X,Y,XDIMS,YDIMS) computes the contracted product of tensors X
%  and Y in the modes specified by the row vectors XDIMS and
%  YDIMS. The sizes in the modes specified by XDIMS and YDIMS must
%  match, i.e. size(X,XDIMS) == size(Y,YDIMS).
% 
%  TTT(X,Y,DIMS) is equivalent to calling TTT(X,Y,DIMS,DIMS).
%
%  In the case of complex tensors, we take the complex conjugate of X.
%
%  Examples
%  X = rand(4,2,3);
%  Y = rand(3,4,2);
%  Z = ttt(X,Y) %<-- outer product of X and Y
%  Z = ttt(X,X,1:3) %<-- inner product of X with itself
%  Z = ttt(X,Y,[1 2 3],[2 3 1]) %<-- inner product of X & permuted Y
%  Z = ttt(X,Y,[1 3],[2 1]) %<-- product of X & Y along specified dims
% 
%  See also TTM.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

% Check input arguments
if(nargin < 2)
  error('Requires at least two arguments.');
elseif(nargin == 2)
  ind_x = [];
end

if(nargin <= 3)
  ind_y = ind_x;
end

% Check x, y
if(~isnumeric(x) || ~isnumeric(y))
  error('First two arguments must be MATLAB (multidimensional) arrays.');
end

% Check ind_x, ind_y
if( ~isindexvector(ind_x) || numel(unique(ind_x)) < numel(ind_x) || ...
    ~isindexvector(ind_y) || numel(unique(ind_y)) < numel(ind_y) )
  error(['IND_X and IND_Y must be vectors of positive integers,' ...
	 ' without double entries.']);
end

% This may help the performance of matricize; the order of the
% dimensions in ind_x, ind_y doesn't matter otherwise
[ind_x, sort_idx] = sort(ind_x);
ind_y = ind_y(sort_idx);

% Calculate effective tensor order
d_x = max([ndims(x) max(ind_x)]);
d_y = max([ndims(y), max(ind_y)]);

% Determine sizes of tensors (with trailing singletons)
sz_x = size(x);
sz_x(end+1:d_x) = 1;
sz_y = size(y);
sz_y(end+1:d_y) = 1;

% Compare dimensions to reduce
if(~isequal(sz_x(ind_x), sz_y(ind_y)))
  error('Tensor dimensions must agree.');
end

% Determine dimensions which are not eliminated
ind_compl_x = setdiff(1:d_x, ind_x);
ind_compl_y = setdiff(1:d_y, ind_y);

% Matricize both tensors
if(ismember(1, ind_x))
  transX = true;
  X = matricize(x, ind_x, ind_compl_x);
else
  transX = false;
  X = matricize(x, ind_compl_x, ind_x);
end

if(ismember(1, ind_y))
  transY = false;
  Y = matricize(y, ind_y, ind_compl_y);
else
  transY = true;
  Y = matricize(y, ind_compl_y, ind_y);
end


% Calculate matricization of z, Z = X'*Y
if(transX)
  if(~transY)
    Z = X'*Y;
  else
    Z = X'*Y.';
  end
else
  if(~transY)
    Z = conj(X)*Y;
  else
    Z = conj(X)*Y.';
  end
end

% Determine size of z and dematricize
sz_z = [sz_x(ind_compl_x), sz_y(ind_compl_y)];

% At least two entries in sz_z (singleton dimensions)
sz_z(end+1:2) = 1;

% Dematricize matrix Z to tensor z
z = dematricize(Z, sz_z, 1:numel(ind_compl_x));