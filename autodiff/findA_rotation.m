function A = findA_rotation(problem)
    name_rotation = problem.M.name();
    indexA = strfind(name_rotation,'indices ') + 8;
    A = str2double(name_rotation(indexA));
    for i = indexA+3:3:length(problem.M.name())
        A = [A,str2double(name_rotation(i))];
    end
end