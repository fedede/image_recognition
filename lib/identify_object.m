function [BW] = identify_object(BW)
%identify_object returns a logical mask with the object in the image marked
%with the positive label
if sum(BW(:)) > (numel(BW) / 2)
        BW = ~BW;
end
end

