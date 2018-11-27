function [area,perimeter,box_area,box_perimeter] = get_object_props(mask)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
props = regionprops(mask, 'FilledArea','Perimeter', 'BoundingBox');
area = 0;
perimeter = 0;
box_area = 0;
box_perimeter = 0;
for i=1:size(props,1)
    area = area + props(i).FilledArea;
    perimeter = perimeter + props(i).Perimeter;
    box_area = box_area + props(i).BoundingBox(3) * props(i).BoundingBox(4);
    box_perimeter = box_perimeter + 2 * props(i).BoundingBox(3) + 2 * props(i).BoundingBox(4);
    
end
end

