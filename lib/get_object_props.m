function [area,perimeter] = get_object_props(mask)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
props = regionprops(mask, 'FilledArea','Perimeter');
area = 0;
perimeter = 0;
for i=1:size(props,1)
    area = area + props(i).FilledArea;
    perimeter = perimeter + props(i).Perimeter;
end
end

