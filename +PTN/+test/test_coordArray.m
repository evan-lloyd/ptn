function [  ] = test_coordArray( sc, nSamples, target, islog )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    if nargin < 2
        nSamples = 1000;
    end
    if nargin < 3
        target = 1;
    end
    
    if nargin < 4
        islog = false;
    end
    
    d = sc{1}.d;
    s = zeros(nSamples, 1);
    for i=1:nSamples
        idx = num2cell(round( (sc{1}.n(1)-1) * (rand(1,d)) ) + 1);
        x = cellfun(@(x) x(idx{:}), sc);
        
        if islog
            s(i) = sum(exp(x(:)));
        else
            s(i) = sum(x(:));
        end
    end
    
    max(abs(s(:)-target))
end

