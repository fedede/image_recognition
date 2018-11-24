function [] = visualize_features(features, labels, f1,f2)
%visualize_features Plots the relation between features given by f1 and f2
label_names = {'Colour feature', 'Texture feature','Shape feature'}; 
gscatter(features(:,f1),features(:,f2),labels)
title('Scatter plot of extracted feature values');
xlabel(label_names{f1})
ylabel(label_names{f2})
end

