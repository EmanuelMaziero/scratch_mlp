% ----------------------------------------------------------------------- %
% Calcula o erro pela função categorical cross-entropy.                   %
%                                                                         %
% Entradas:                                                               %
%   y - saídas preditas                                                   %
%   y_hat - saídas desejadas                                              %
%                                                                         %
% Saídas:                                                                 %
%   ce - erro pela função categorical cross-entropy                       % 
% ----------------------------------------------------------------------- %

function ce = cross_entropy(y, y_hat)
    % Compara as saídas da softmax com o vetor onehot desejado e faz a
    % média entre os 10 erros obtidos.
    ce = -1*mean(y_hat.*log(y), 'all');
end