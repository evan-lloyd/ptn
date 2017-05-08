function [d] = KL(p,q)
% Get KL divergence between distributions (as vectors) p and q.
% (Assumes vectors are normalized; nudge zeros to a small epsilon to
% make this always defined).
    epsilon = 1e-14;
    q(q <= 0) = epsilon;
    p(p <= 0) = epsilon;
    d = p .* (log(p) - log(q));
    d = sum(d(:));
end