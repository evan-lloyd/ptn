function [ a ] = delta(d, n)
% delta(d,n) -- make a delta tensor of order d and extent n^d
%   delta(d,n)(i_1,i_2...i_d) = { 1 if i_1=i_2=...=i_d
%                               { 0 otherwise
    s = repmat(n, [1 d]);
    a = zeros(s);
    for i=1:n
        idx = repmat({i}, [d 1]);
        a(idx{:}) = 1;
    end

end

