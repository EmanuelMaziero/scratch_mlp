% ----------------------------------------------------------------------- %
% Função de ativação softmax. Recebe N entradas e atribui uma porcentagem %
% a cada uma delas, fazendo com que a soma de todas seja 1. Aqui,         %
% atribui a probabilidade daquela ser a imagem predita correta.           %
%                                                                         %
% Entradas:                                                               %
%   x - saída dos neurônios da camada de saída                            %
%                                                                         %
% Saídas:                                                                 %
%   y - saída da rede neural                                              % 
% ----------------------------------------------------------------------- %

function y = softmax(x)
    e = 0;
    y = zeros(10,1);
    % Somatório da exponencial de todas entradas
    for i=1:length(x)
        e = e + exp(x(i));
    end
    % Cálculo da softmax
    for j=1:length(x)
        y(j) = exp(x(j))/e;
    end
end