function y = full(x)
%FULL Convert an htensor to a (dense) tensor.
%
%   Y = FULL(X) converts an htensor to a (dense) tensor.
%
%   Examples
%   X = htenrandn([2 4 3 5]);
%   Y = full(X) %<-- equivalent dense tensor
%
%   See also HTENSOR, TENSOR.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

% Left singular vectors at each node
U = cell(1, x.nr_nodes);

% Go through all nodes, starting at the leafs
for ii=x.nr_nodes:-1:1
  
  if(x.is_leaf(ii))
    % Left singular vectors already known
    U{ii} = x.U{ii};
  else
    % Find child nodes
    ii_left  = x.children(ii, 1);
    ii_right = x.children(ii, 2);

    BUU = ttm(x.B{ii}, {U{ii_left}, U{ii_right}}, [1 2]);
    U{ii} = matricize(BUU, [1 2]);
    
    % Clear variables to save memory
    clear BUU;
    U{ii_left} = [];
    U{ii_right} = [];
    
  end 
end

% Vectorization of the full tensor is in U{1},
% Reshape to get full tensor
y = dematricize(U{1}, size(x), x.dims{1}, []);


%----------------------------------------------------------------%
% Evaluates one element of htensor x, indicated by ind. This     %
% function is only provided for illustration purposes, it is     %
% equivalent to full_block(x, ind)                                %
%----------------------------------------------------------------%
function x_ind = eval_element(x, ind)

U = cell(1, x.nr_nodes);

for ii=x.nr_nodes:-1:1
  
  if(x.is_leaf(ii))
    U{ii} = x.U{ii}(ind(x.dims{ii}), :);
  else
    ii_left  = x.children(ii, 1);
    ii_right = x.children(ii, 2);
    
    U{ii} = zeros(1, x.rank(ii));

    % Slow method
    %for jj=1:x.rank(ii)
    %  for kk=1:x.rank(ii_left)
    %    for ll=1:x.rank(ii_right)
    %      U{ii}(jj) = U{ii}(jj) + ...
    %          x.B{ii}(kk, ll, jj) * ...
    %      kron(U{ii_right}(:, ll), U{ii_left}(:, kk));
    %    end
    %  end
    %end
    
    % faster method
    %for jj=1:x.rank(ii)
    %  U{ii}(jj) = U{ii_left}*...
    %	  x.B{ii}(:, :, jj)*U{ii_right}.';
    %end
    
    % final version
    BUU = ttm(x.B{ii}, {U{ii_left}, U{ii_right}}, [1 2]);
    U{ii} = matricize(BUU, [1 2]);
    
  end 
end

x_ind = U{1};