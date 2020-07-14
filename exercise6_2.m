load digit.mat X T;
[d, nX, nc] = size(X);
nT = size(T, 2);

% initialize
S = zeros(d, d);
mu = zeros(d, nc);
p = zeros(nc, nT, nc);
C = zeros(nc, nc);

% 標本平均と分散共分散行列の推定
for c = 1 : nc
    mu(:, c) = mean(X(:, :, c), 2);
    S = S + cov(X(:, :, c)') / nc;
end

% 事後確率の計算
for ct = 1 : nc
    for c = 1 : nc
        muc = mu(:, c);
        t = T(:, :, ct);
        % inv(S) * muc -> S \ muc;
        p(ct, :, c) = t' * (S \ muc) - muc' * (S \ muc) / 2;
    end
end

% 事後確率が最大のカテゴリに属するとして数え上げ
[pmax, P] = max(p, [], 3);
for ct = 1 : nc
    for c = 1 : nc
        C(ct, c) = sum(P(ct, :) == c);
    end
end

% 誤識別率の計算
pe = diag(C) / nT;
disp(pe);