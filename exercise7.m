n = 100; % 人工データの数
h = 0.5;

% p(x): 混合正規分布
mu = [0; 7];
sigma(1,1,:) = [1^1; 2^2];
gm = gmdistribution(mu, sigma);

rng('default'); % 乱数生成の初期化
X = gm.random(n); % 混合正規分布に従う乱数列の生成(トレーニングデータ)
XT = gm.random(n); % テストデータ

fn_hp = generate_hp(n, X, h);

% 真の分布
fplot(@(x) transpose(gm.pdf(transpose(x))), [-5 15], 'LineWidth', 2);
hold on;
% 推定した分布
fplot(fn_hp, [-5 15], 'LineWidth', 2);
leg = legend({'$ p(x) $','$ \hat{p}(x) $'}, 'Location', 'northeast');
set(leg, 'Interpreter', 'latex');
set(leg, 'FontSize', 13);
hold off;

% K(x): ガウスカーネル
function [ret] = k(x)
    ret = 1 ./ sqrt(2 * pi) .* exp(x.^2 ./ (-2));
end

function [ret] = generate_hp(n, X, h)
    ret = @(x) sum(k(((x .* ones(n, 1)) - X) ./ h)) / (n * h);
end