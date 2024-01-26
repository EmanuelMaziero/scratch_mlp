% ----------------------------------------------------------------------- %
% Derivada da função categorical cross-entropy, necessária para o         %
% backpropagation.                                                        %
%                                                                         %
% Entradas:                                                               %
%   y - saídas preditas                                                   %
%   y_hat - saídas desejadas                                              %
%                                                                         %
% Saídas:                                                                 %
%   dce - dE/dY da camada de saída                                        % 
% ----------------------------------------------------------------------- %

function dce = dcross_entropy(y, y_hat)
    dce = y - y_hat;
end