% ----------------------------------------------------------------------- %
% Função de ativação sigmoid.                                             %
%                                                                         %
% Entradas:                                                               %
%   x - saída do neurônio                                                 %
%                                                                         %
% Saídas:                                                                 %
%   sigm - saída do neurônio submetida a sigmoid                          % 
% ----------------------------------------------------------------------- %

function sigm = sigmoid(x)
    sigm = 1./(1+exp(-x));
end