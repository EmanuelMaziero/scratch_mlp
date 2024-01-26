% ----------------------------------------------------------------------- %
% Essa função usa o erro calculado e realiza a regra da cadeia para       %
% calcular as equações do backpropagation e obter os fatores de           %
% atualização dos pesos e bias.                                           %
%                                                                         %
% Entradas:                                                               %
%   w - struct de pesos                                                   %
%   b - struct de bias                                                    %
%   a - struct com as saídas z submetidas a função de ativação            %
%   alpha - passo de aprendizado                                          %
%   in - imagem vetorizada que entra na rede                              %
%   onehot - vetor de predições codificado para uso da softmax            %
%   j - variável que percorre as imagens                                  %
%                                                                         %
% Saídas:                                                                 %
%   w - struct com os pesos atualizados pelo backpropagation              %
%   b - struct com os bias atualizados pelo backpropagation               %   
% ----------------------------------------------------------------------- %

function [w, b] = backprop(w, b, a, alpha, in, onehot, j)
    % Gradientes de entrada das camadas
    d_hid3 = dcross_entropy(a.out, onehot(:,j));
    d_hid2 = (w.out.'*d_hid3).*dsigmoid(a.hid2);
    d_hid1 = (w.hid2.'*d_hid2).*dsigmoid(a.hid1);
    
    % Gradientes dos pesos e bias das camadas
    dw_out = d_hid3*a.hid2.';
    db_out = d_hid3;
    dw_hid2 = d_hid2*a.hid1.';
    db_hid2 = d_hid2;
    dw_hid1 = d_hid1*in.';
    db_hid1 = d_hid1;

    % Atualização dos parâmetros (descida do gradiente)
    w.out = w.out - (alpha*dw_out);
    w.hid2 = w.hid2 - (alpha*dw_hid2);
    w.hid1 = w.hid1 - (alpha*dw_hid1);
    b.out = b.out - (alpha*db_out);
    b.hid2 = b.hid2 - (alpha*db_hid2);
    b.hid1 = b.hid1 - (alpha*db_hid1);
end