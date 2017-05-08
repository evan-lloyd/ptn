function [a]=tt_sum(varargin)
%A=B+C
%   [A]=PLUS(B,C)  Adds two TT-tensors B and C
%
%
% TT-Toolbox 2.2, 2009-2012
%
%This is TT Toolbox, written by Ivan Oseledets et al.
%Institute of Numerical Mathematics, Moscow, Russia
%webpage: http://spring.inm.ras.ru/osel
%
%For all questions, bugs and suggestions please mail
%ivan.oseledets@gmail.com
%---------------------------

tensorFilter = cellfun(@ (x) x.d, varargin);
tensors = varargin(tensorFilter > 0);

a=tt_tensor;

if numel(tensors) == 0
    return;
end

n = tensors{1}.n;
d = tensors{1}.d;

if d==1
    a = tensors{1};
    cores = cellfun(@(x) x.core, tensors, 'UniformOutput', false);
    a.core = sum(cat(3, cores{:}), 3);
    return;
end

r= sum(cell2mat(cellfun(@(x) x.r, tensors, 'UniformOutput', false)), 2);  %ra+rb; 
r(1) = tensors{1}.r(1);
r(d+1) = tensors{1}.r(d+1);
a.d=d;
a.n=n;
a.r=r;

sz=dot(n.*r(1:d),r(2:d+1));
pos=(n.*r(1:d)).*r(2:d+1);
pos=cumsum([1;pos]);
a.ps=pos;
cr=zeros(sz,1);

% The result cores are in fact block diagonal, if we reshape all the
% component cores to matrices of the right size.
dimCell = num2cell(2:d-1);
cores = cellfun(@(i) getdiag(cellfun(@(x) sparse(reshape(x.core(x.ps(i):x.ps(i+1)-1), x.r(i), n(i)*x.r(i+1))), ...
                                  tensors, 'UniformOutput', false)),...
    dimCell, 'UniformOutput', false);
cores = cellfun(@(x) x(:), cores, 'UniformOutput', false);

cr(pos(2):pos(d)-1) = cat(1, cores{:});

% first dim is a special case; instead of block diag we stack horizontally
firstCores = cellfun(@(x) reshape(x.core(x.ps(1):x.ps(2)-1), x.r(1), n(1) * x.r(2)), tensors, 'UniformOutput', false);
cr3 = cat(2, firstCores{:});
cr(pos(1):pos(2)-1) = cr3(:);

% last dim is a special case; stack vertically
lastCores = cellfun(@(x) reshape(x.core(x.ps(d):x.ps(d+1)-1), x.r(d), n(d) * x.r(d+1)), tensors, 'UniformOutput', false);
cr3 = cat(1, lastCores{:});
cr(pos(d):pos(d+1)-1) = cr3(:);

a.core=cr;
return
end

function [d] = getdiag(cores)
    d = blkdiag(cores{:});
end