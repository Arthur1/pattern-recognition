n = 100; % 人工データの数
vec_h = [0.1; 0.5; 1; 2; 4; 8]; % hの幅

% p(x): 混合正規分布
mu = [0; 7];
sigma(1,1,:) = [1^1; 2^2];
gm = gmdistribution(mu, sigma);

rng('default'); % 乱数生成の初期化
X_train = gm.random(n); % 混合正規分布に従う乱数列の生成(トレーニングデータ)
X_test = gm.random(n); % テストデータ

% hごと対数尤度の平均の計算
A = zeros(length(vec_h), 1); % トレーニングデータに対する対数尤度の平均
B = zeros(length(vec_h), 1); % テストデータに対する対数尤度の平均
fprintf("     h        A         B\n");
for j = 1 : length(vec_h)
    A(j) = lnLH(X_train, X_train, vec_h(j));
    B(j) = lnLH(X_train, X_test, vec_h(j));
    disp([vec_h(j), A(j), B(j)]);
end

% Bが最大のモデルを描画
[~, maxj] = max(B);
fn_hp = KDE(X_train, vec_h(maxj));
% 真の分布
fplot(@(x) transpose(gm.pdf(transpose(x))), [-5 15], 'LineWidth', 2);
hold on;
% 推定した分布
fplot(fn_hp, [-5 15], 'LineWidth', 2);
leg = legend({'$ p(x) $','$ \hat{p}(x) $'}, 'Location', 'northeast');
set(leg, 'Interpreter', 'latex');
set(leg, 'FontSize', 13);
ttl = title(sprintf('$\\hat{h} = %.1f$', vec_h(maxj)));
set(ttl, 'Interpreter', 'latex');
set(ttl, 'FontSize', 16);
hold off;

% カーネル密度推定
function fn = KDE(X, h)
    n = length(X);
    % ガウスカーネルを利用
    fn_k = @(x) 1 ./ sqrt(2 * pi) .* exp(x.^2 ./ (-2));
    function ret = p(x)
        vec_x = x .* ones(n, 1);
        ret = sum(fn_k((vec_x - X) ./ h)) / (n * h);
    end
    fn = @p;
end

% log-sum-exp
function ret = logSumExp(vec)
    maxV = max(vec);
    ret = maxV + log(sum(exp(vec - maxV)));
end

% ln p^(x)
function fn = gen_lnph(X_train, h)
    n = length(X_train);
    function ret = lnph(x)
        vec_tmp = zeros(n, 1);
        for i = 1 : n
            vec_tmp(i) = - ((x - X_train(i)) / h)^2 / 2;
        end
        ret = logSumExp(vec_tmp) - log (n * h) - log(2 * pi) / 2;
    end
    fn = @lnph;
end

% ln LH = Σ ln p^(x_j)
function ret = lnLH(X_train, X, h)
    lnph = gen_lnph(X_train, h);
    m = length(X);
    vec_tmp = zeros(m, 1);
    for j = 1 : m
        vec_tmp(j) = lnph(X(j));
    end
    ret = mean(vec_tmp);
end
