% ----------------------------------------------------------------------- %
% Essa função recebe uma entrada (imagem) e realiza a predição pela       %
% função softmax.                                                         %
%                                                                         %
% Entradas:                                                               %
%   X - imagem vetorizada                                                 %
%   w - struct de pesos                                                   %
%   b - struct de bias                                                    %
%   z - struct com as saídas dos neurônios                                %
%   a - struct com as saídas z submetidas a função de ativação            %
%                                                                         %
% Saídas:                                                                 %
%   z - struct atualizada com as saídas dos neurônios                     %
%   a - struct atualizada com as saídas z submetidas a função de ativação %
%   in - imagem vetorizada que entra na rede                              %     
% ----------------------------------------------------------------------- %

function [a, z, in] = forprop(X, w, b, a, z)
    % Vetorização da entrada
    in = reshape(X, [784 1]);

    % Camada de entrada <-> Camada escondida 1
    % 100x784 * 784x1 + 100x1 = 100x1
    z.hid1 = (w.hid1*in) + b.hid1;
    % Ativação
    a.hid1 = sigmoid(z.hid1);     

    % Camada escondida 1 <-> Camada escondida 2
    % 100x100 * 100x1 + 100x1 = 100x1
    z.hid2 = (w.hid2*a.hid1) + b.hid2;
    % Ativação
    a.hid2 = sigmoid(z.hid2);    

    % Camada escondida 2 <-> Camada de saída
    % 10x100 * 100x1 + 10x1 = 10x1
    z.out = (w.out*a.hid2) + b.out;  
    % Ativação
    a.out = softmax(z.out);      
end