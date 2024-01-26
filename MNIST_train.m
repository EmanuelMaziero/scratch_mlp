% Predição de imagens de número escritos a mão (MNIST dataset).
% Treinamento.
clear
clc
 
% Carrega um arquivo .csv contendo os dígitos e suas etiquetas
dataset = load('mnist_train.csv');

% Número de imagens
n_imgs = size(dataset,1);

% Separação das etiquetas em uma variável
labels = dataset(1:n_imgs, 1);

% Separação das imagens em uma variável
imgs = dataset(1:n_imgs, 2:785);

% Normalização dos dados
imgs = ((imgs-min(imgs,[],'all'))/(max(imgs,[],'all')-min(imgs,[],'all')));

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

% Iterações
epochs = 10;

% Passo de aprendizagem
alpha = 0.0058;

% Número de neurônios nas camadas
n = struct();
% A imagem 28x28 entra vetorizada em um vetor de 28x28 = 784 pixels
n.in = size(imgs, 2);
n.hid1 = 100;
n.hid2 = 100;
% Usando o MNIST, temos que ter 10 saídas, 1 para cada dígito
n.out = 10;

% Bias (valores aleatórias de -1 a 1)
b = struct();
b.hid1 = rand(n.hid1, 1)*2-1;
b.hid2 = rand(n.hid2, 1)*2-1;
b.out = rand(n.out, 1)*2-1;

% Pesos (valores aleatórias de -1 a 1)
w = struct();
w.hid1 = rand(n.hid1, n.in)*2-1;
w.hid2 = rand(n.hid2, n.hid1)*2-1;
w.out = rand(n.out, n.hid2)*2-1;

% 'z' para a saída do neurônio
% 'a' para a função de ativação em 'z'
z = struct();
a = struct();

% Pré-alocação
accuracy = zeros(1,10);
e = zeros(10, n_imgs);

% Alterna as épocas
for i=1:epochs
    count = 0;
    % Alterna as imagens
    for j=1:n_imgs
        % Forward Propagation
        X = imgs(j,:);
        [a, z, in] = forprop(X, w, b, a, z);

        % Cálculo do Erro pela função Cross-Entropy
        e(i,j) = cross_entropy(a.out, onehot(:,j));

        % Backward Propagation
        [w, b] = backprop(w, b, a, alpha, in, onehot, j);
        
        % A variável 'count' recebe o valor da função 'acuracia',
        % que calcula quantas imagens foram peditas corretamente.
        count = acuracia(onehot(:,j), a, count);
    end
    accuracy(i) = count;
end

% Calcula a porcentagem total de imagens preditas corretamente
last_epoch_accuracy = accuracy(10)/n_imgs;

% Calculo do erro médio em cada época
erro = zeros(1,10);
for n=1:size(e,1)
    erro(n) = mean(e(n,:));
end

% Plotagem do erro e da acurácia
figure
plot(100*(accuracy./n_imgs), 'o-')
xlabel('Época')
ylabel('%')
grid on
title('Acurácia')

figure
plot(10*log10(erro), 'o-')
xlabel('Época')
ylabel('dB')
grid on
title('Erro')

text1 = ['A acurácia final obtida no treinamento da rede neural foi de: ', num2str(100*(accuracy(end)/n_imgs)), '%'];
disp(text1)
text2 = ['O erro final obtido no treinamento da rede neural foi de: ', num2str(erro(end))];
disp(text2)

% Salva os parâmetros treinados
wo = w.out;
wh2 = w.hid2;
wh1 = w.hid1;
bo = b.out;
bh2 = b.hid2;
bh1 = b.hid1;
save('w_out.mat','wo');
save('w_2.mat'  ,'wh2');
save('w_1.mat'  ,'wh1');
save('b_out.mat','bo');
save('b_2.mat'  ,'bh2');
save('b_1.mat'  ,'bh1');