function out = subsref(x, s)
%SUBSREF Subscripted reference for htensor.
%
%   Examples
%   x = htensor([5 9 3])
%   x.children returns a nr_nodes x 2 matrix specifying the
%                  dimension tree.
%   x.U returns a cell array of X.nr_nodes matrices.
%   x.B{1} returns the tensor corresponding to the root node.
%   x{ii} returns x.U{ii} for leaf node ii, and x.B{ii} otherwise.
%   x(2,3,1) calculates and returns that single element of x.
%   x(1,2:3,:) returns an htensor with these elements of x.
%   x(:) returns the vectorization of x.
%
%   See also HTENSOR.
%

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

switch s(1).type
  
 case '.'
  
  out = builtin('subsref', x, s);
  
 case '()'
  % x(:)
  if(length(s(1).subs) == 1 && ischar(s(1).subs{1}))
    out = full(x);
    out = out(:);
  % x([1 3 4], :, 2, 1:3)
  elseif(length(s(1).subs) == ndims(x))
    
    ind = s(1).subs;
    
    for ii=find(x.is_leaf)
      % Restrict leaf matrix to the elements given by ind
      d_ii = x.dims{ii};
      if( ~ischar(ind{d_ii}) && max(ind{d_ii}) > size(x.U{ii}, 1) )
        error('Index exceeds matrix dimensions.');
      end
      if( ~ischar(ind{d_ii}) && any(ind{d_ii} <= 0) )
        error('Subscript indexes must be real positive integers.');
      end
      x.U{ii} = x.U{ii}(ind{d_ii}, :);
    end    
    
    if(all(size(x) == 1))
      out = full(x);
    else
      out = x;
    end
  else
    error('Number of arguments does not match htensor size.');
  end  
  
  
 case '{}'
  
  if(length(s(1).subs) ~= 1 || ~isnumeric(s(1).subs{1}) || ...
     s(1).subs{1} <= 0)
    error('{} only takes one positive integer.');
  end
  
  ii = s(1).subs{1};
  if(ii > x.nr_nodes)
    error('Index exceeds number of nodes in dimension tree.');
  end
  
  if(x.is_leaf(ii))
    out = builtin('subsref', x.U, s);
  else
    out = builtin('subsref', x.B, s);
  end
  
end
