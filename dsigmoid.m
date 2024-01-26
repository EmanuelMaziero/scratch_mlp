% ----------------------------------------------------------------------- %
% Derivada da Função de ativação sigmoid, necessária no backpropagation.  %
%                                                                         %
% Entradas:                                                               %
%   x - saídas dos neurônios submetidas a função de ativação.             %
%                                                                         %
% Saídas:                                                                 %
%   dsigm - usada no cálculo dos dE/dX                                    % 
% ----------------------------------------------------------------------- %

function dsigm = dsigmoid(x)
    dsigm = x.*(1-x);
end