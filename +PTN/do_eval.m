function [ phi ] = do_eval( net, parms, order )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    % validate tn structure and parameters
    %for i=1:net.n
    %    if (size(parms{i},2) == 1 && parms{i}.d ~= net.a(i) || any(parms{i}.n' ~= net.d{i})
    %        ['Error at parameter ' num2str(i) ': ' num2str(parms{i}.n') ', ' num2str(net.d{i})]
    %        error('Network structure inconsistent with shapes of given parameters');
    %    end
    %end
    if nargin < 3
        order = arrayfun(@(x) {x}, 1:net.n, 'UniformOutput', false);
    end
    [~, phi] = PTN.tn_eval(net, parms, order);
    phi = phi{1};
end

