realdata=load(fullfile('.','results','Aufnahmen_OCM_Band_Kante_Halbkugeln_Halbkugeln_kante_1_COMP.csv_Tracks.mat'));
figure(1),clf,hold on
for i=1:numel(realdata.particles)
%     plot(realdata.particles{i}(:,1),realdata.particles{i}(:,2))
    plot(realdata.particles{i}(2:end,2),diff(realdata.particles{i}(:,2)));
    if i==10
        break
    end
end

xlabel('Pos on belt in transport direction in px');
ylabel('Velocity in px / time step');