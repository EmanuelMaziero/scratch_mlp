% ----------------------------------------------------------------------- %
% Função de ativação sigmoid.                                             %
%                                                                         %
% Entradas:                                                               %
%   onehot - vetor de predições codificado para uso da softmax            %
%   a - struct com as saídas z submetidas a função de ativação            %
%   count - pré-alocação do contador                                      %
%                                                                         %
% Saídas:                                                                 %
%   count - conta quantos imagens foram preditas corretamente             %
% ----------------------------------------------------------------------- %

function count = acuracia(onehot, a, count)
    % Cálculo o maior valor do vetor (maior probabilidade) e armazena o
    % índice dele.
    [v1, ind1] = max(onehot);
    correct = ind1;
    [v2, ind2] = max(a.out);
    guess = ind2;

    % Compara o índice armazenada pela predição e pela saida desejada.
    % Se eles forem iguais, a predição foi correta.
    if correct == guess
        count = count +1;
    end
end