% Predição de imagens de número escritos a mão (MNIST dataset).
% Validação.
clear
clc
 
% Carrega um arquivo .csv contendo os dígitos e suas etiquetas
dataset = load('mnist_test.csv');

% Número de imagens
n_imgs = 10000;

% Separação das etiquetas em uma variável
labels = dataset(1:n_imgs, 1);

% Separação das imagens em uma variável
imgs = dataset(1:n_imgs, 2:785);

% Normalização dos dados
imgs = ((imgs-min(imgs,[],'all'))/(max(imgs,[],'all')-min(imgs,[],'all')));

% Carrega os parâmetros treinados
w = struct();
b = struct();

load('w_out.mat');
w.out = wo;
load('w_2.mat');
w.hid2 = wh2;
load('w_1.mat');
w.hid1 = wh1;
load('b_out.mat');
b.out = bo;
load('b_2.mat');
b.hid2 = bh2;
load('b_1.mat');
b.hid1 = bh1;

% Codificação ONE-HOT ENCODING
% Os dígitos vão de 0 a 9, então teremos 10 codificações para cada etiqueta
onehot = zeros(10,n_imgs);

% Percorrendo cada etiqueta, a variável 'onehot', de shape (10, n_imgs),
% recebe o valor da etiqueta na linha 'labels(l)+1' (+1 para compensar o 
% fato de não termos índice 0 no MATLAB) e o valor de 'l' na coluna
% de forma a posicionar o valor 1 na posição do valor da etiqueta+1
for l=1:length(labels)
    % Ex.: onehot(labels(1)+1,1) = 1
    %      onehot(2+1,1) = 1
    onehot(labels(l)+1,l) = 1;
end

% 'z' para a saída do neurônio
% 'a' para a função de ativação em 'z'
z = struct();
a = struct();

% Pré-alocação
e = zeros(10, n_imgs);

count = 0;
% Alterna as imagens
for j=1:n_imgs
    % Forward Propagation
    X = imgs(j,:);
    [a, z, in] = forprop(X, w, b, a, z);

    % Cálculo do Erro pela função Cross-Entropy
    e(j) = cross_entropy(a.out, onehot(:,j));
    
    % A variável 'count' recebe o valor da função 'acuracia',
    % que calcula quantas imagens foram peditas corretamente.
    count = acuracia(onehot(:,j), a, count);
end

% Calcula a porcentagem total de imagens preditas corretamente
final_accuracy = sum(count)/(n_imgs);
text1 = ['A acurácia obtida na validação da rede neural foi de: ', num2str(100*final_accuracy), '%'];
disp(text1)
final_error = mean(e, 'all');
text2 = ['O erro final obtido na validação da rede neural foi de: ', num2str(100*final_error)];
disp(text2)