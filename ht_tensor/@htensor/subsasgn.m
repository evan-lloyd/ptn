function x = subsasgn(x, s, v)
%SUBSASGN Subscripted assignment for an htensor.
%
%   Examples
%   x = htensor([5 9 3])
%   x.U{2}       = U_new    % dimensions of U_new must match old x.U{2}
%   x.U{2}(5, 2) = 7
%   x.B{1}       = B_new    % dimensions of B_new must match old x.B{1}
%   x{ii}        = UB_new   % assigns to x.U{ii} if ii is a leaf node, or
%                           % to x.B{ii} otherwise (dimensions must
%                           % match).
%   NOT POSSIBLE:
%   x(2,3,1) = 3            % direct assignment of an element
%
%   See also HTENSOR, HTENSOR/SUBSREF.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

switch s(1).type
  
 case '.'
  
  % x.U{ii} = U_new;
  if( s(1).subs == 'U' )
    
    % This allows for x.U{2}(4, 5) = 9, too:
    U_new = x.U;
    U_new = builtin('subsasgn', U_new, s(2:end), v);
    
    for ii=1:x.nr_nodes
      if( any(size(U_new{ii}) ~= size(x.U{ii})) || ...
	  ~isnumeric(U_new{ii}) )
	error('Incompatible type or dimensions in assignment.')
      end
    end
    
    x.U = U_new;
    
  % x.B{ii} = B_new;
  elseif( s(1).subs == 'B' )

    B_new = x.B;
    B_new = builtin('subsasgn', B_new, s(2:end), v);
    
    for ii=1:x.nr_nodes
      if( ~isequal(size(B_new{ii}), size(x.B{ii})) || ...
	  ~isnumeric(B_new{ii}) )
	error('Incompatible type or dimensions in assignment.')
      end
    end
    
    x.B = B_new;
    
  else
    error('Cannot change field %s, field may not exist.', s(1).subs);
  end
    
 case '()'
  error('Cannot change individual entries in an htensor.')
  
  % x{ii} = B_new; (or U_new for leaf nodes ii)
 case '{}'

  if(length(s(1).subs) ~= 1)
    error('htensor.subsasgn: {} only takes one positive integer.')
  end
  
  ii = s(1).subs{1};
  
  s_new.type = '.';
  
  if(x.is_leaf(ii))
    s_new.subs = 'U';
  else
    s_new.subs = 'B';
  end
  
  x = subsasgn(x, [s_new, s], v);

end
