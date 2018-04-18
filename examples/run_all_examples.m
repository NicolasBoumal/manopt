
list = ls('*.m');

for k = 1 : size(list, 1)
    s = strtrim(list(k, :));
    eval([s(1:end-2), ';']);
end
