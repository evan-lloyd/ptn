function G = gramians_sum(x_cell)
%GRAMIANS_SUM Gramians of matricization at each node of a sum of htensors.
%
%   G = GRAMIANS_SUM(X) returns a cell-array containing the reduced
%   gramians G_t = Y_t*Y_t'.
%
%   Node t's matricization of X can be written as
%     X_t = U_t * Y_t,
%   where U_t is assembled from the descendants of node t. The
%   Gramian of the t-unfolding of X is
%   X_t * X_t' = U_t * G_t * U_t'.
%
%   See also GRAMIANS, GRAMIANS_NONORTHOG, HTENSOR

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

x = x_cell{1};

s = numel(x_cell);

% Check size(x_cell): [1 s] not [s 1] !:
if(size(x_cell, 1) ~= 1 && size(x_cell, 2) == 1)
  x_cell = transpose(x_cell);
elseif(size(x_cell, 1) ~= 1)
  error('X_CELL must be a vector cell array.')
end

% Calculate M{ii} = x.U{ii}'*x.U{ii}:
M = cell(1, x.nr_nodes);

% Start at leaves, move up the levels
for ii=x.nr_nodes:-1:2
  
  if(x.is_leaf(ii))
    % Concatenate x_cell{jj}.U{ii} for jj=1:s:
    x.U{ii} = cell2mat(cellfun(@(t)(t.U{ii}), x_cell, ...
			       'UniformOutput', false));
    
    % Block structure of x.U{ii}:
    k = cell2mat(cellfun(@(t)(size(t.U{ii}, 2)), x_cell, ...
			 'UniformOutput', false));
    
    % M{ii} is a cell array containing the blocks:
    M{ii} = mat2cell(full(x.U{ii}'*x.U{ii}), k, k);
    
  else
    ii_left  = x.children(ii, 1);
    ii_right = x.children(ii, 2);
    
    % cell array of diagonal blocks of x.B{ii}:
    B_cell = cell(s, 1);
    for jj=1:s
      B_cell{jj} = x_cell{jj}.B{ii};
    end
    
    % calculate M{ii}:
    M{ii} = cell(s, s);
    for kk=1:s
      for ll=1:s
        B1_klk = ttm(B_cell{kk}, M{ii_left}{ll, kk}, 1);
        B2_llk = ttm(B_cell{ll}, M{ii_right}{kk, ll}, 2);
    
        M{ii}{kk, ll} = ttt(B1_klk, B2_llk, [1 2]);
      end
    end
    
  end
end

G = cell(1, x.nr_nodes);
G{1} = 1;

% Child nodes
root_left  = x.children(1, 1);
root_right = x.children(1, 2);

% Calculate < B{ii}, B{ii} x_1 G{ii} >_(1, 2) and _(1, 3)
for kk=1:s
  for ll=1:s
    B_mod = ttm(conj(x_cell{ll}.B{1}), M{root_right}{kk, ll}, 2);
    G{root_left}{kk, ll} = ttt(conj(x_cell{kk}.B{1}), B_mod, [2 3]);
    
    B_mod = ttm(conj(x_cell{ll}.B{1}), M{root_left}{kk, ll}, 1);
    G{root_right}{kk, ll} = ttt(conj(x_cell{kk}.B{1}), B_mod, [1 3]);
  end
end

% Start from root node, move down.
for ii=find(x.is_leaf == false)
  
  if(ii == 1)
    continue;
  end
  
  % Child nodes
  ii_left  = x.children(ii, 1);
  ii_right = x.children(ii, 2);
  
  B_cell = cell(s, 1);
  for jj=1:s
    B_cell{jj} = x_cell{jj}.B{ii};
  end
  
  G{ii_left} = cell(s, s);
  for kk=1:s
    for ll=1:s
      B_mod_kkl  = ttm(conj(B_cell{kk}), G{ii}{ll, kk}, 3);
      B_right_lkl = ttm(conj(B_cell{ll}), M{ii_left }{kk, ll}, 1);
      B_left_llk  = ttm(conj(B_cell{ll}), M{ii_right}{kk, ll}, 2);
       
      G{ii_left }{kk, ll} = ttt(B_mod_kkl, B_left_llk , [2 3]);
      G{ii_right}{kk, ll} = ttt(B_mod_kkl, B_right_lkl, [1 3]);
    end
  end
  
end

for ii=2:x.nr_nodes
    G{ii} = cell2mat(G{ii});
end
