function [ out ] = draw_from_JPT( JPT, n )
% Draw a random IID sample from a joint probability tensor (in full format) by
% treating it as a multinomial with the given probabilities. If n > 1
% given, instead generate a vector of counts.
% Relies on the lightspeed routine sample_hist.

    if nargin < 2 || n < 1
        n = 1;
    end

    p = JPT(:);
    p = p / sum(p);

    r = sample_hist(p(:), n);
    out = reshape(r, size(JPT));
    
    return;
end

