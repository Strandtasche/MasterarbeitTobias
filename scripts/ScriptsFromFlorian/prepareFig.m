function prepareFig(size,scaling,fontSize,lineWidth,useLaTeX,useMathAxis)
    % prepareFig v4 by Florian Pfaff pfaff@kit.edu
    %
    % Parameters:
    %   size (1 x 2 vector)
    %       Set the target size of the plot.
    %   scaling (scalar)
    %       Scaling of the plot. This can help, e.g., to prevent large
    %       legend boxes.
    %   fontSize (scalar)
    %       Set the size of the font used.
    %   lineWidth (scalar)
    %       Set the width of the lines of the plot.
    %   useLaTeX (logical)
    %       Choose whether text (axis labels, tick labels, legends) should
    %       be rendered using LaTeX
    %   useMathAxis (logical with up to three entries)
    %       Choose if math mode should be invoked for axis labels. If only
    %       one logical is given, it is enables/disabled for all axes.
    %
    % Changelog:
    % 4.0 Adding $ $ to tick labels when real numbers or \pi is found.
    % 3.1 allows diabling LaTeX
    switch nargin
        case 0
            size=[8,5];
            scaling=2;
            fontSize=7;
            lineWidth=0.6;
            useLaTeX=true;
            useMathAxis=false;
        case 1
            scaling=2;
            fontSize=7;
            lineWidth=0.6;
            useLaTeX=true;
            useMathAxis=false;
        case 2
            fontSize=7;
            lineWidth=0.6;
            useLaTeX=true;
            useMathAxis=false;
        case 3
            lineWidth=0.6;
            useLaTeX=true;
            useMathAxis=false;
        case 4
            useLaTeX=true;
            useMathAxis=false;
        case 5
            useMathAxis=false;
    end
    assert((scaling>0.1)&&(scaling<10));
    assert((fontSize>3)&&(fontSize<20));
    assert((lineWidth>0.1)&&(lineWidth<3));
    assert(islogical(useMathAxis));
    
    switch numel(useMathAxis)
        case 1
            useMathXAxis=useMathAxis;useMathYAxis=useMathAxis;useMathZAxis=useMathAxis;
        case 2
            useMathXAxis=useMathAxis(1);useMathYAxis=useMathAxis(2);useMathZAxis=false;
        case 3
            useMathXAxis=useMathAxis(1);useMathYAxis=useMathAxis(2);useMathZAxis=useMathAxis(3);
    end
    allAxes=findall(gcf, 'Type', 'Axes');
    box on
    set(allAxes,'LineWidth',lineWidth*scaling);
    set(findall(allAxes,'type','line'),'LineWidth',lineWidth*scaling);
    set(allAxes,'Color','none');
    paperSize=size*scaling;
    set(allAxes, 'FontSize', fontSize*scaling);
    allText=findall(gca, 'Type', 'Text');
    set(allText,'FontSize', fontSize*scaling);
    delete(title('')); % Set title blank and delete object (so it won't turn up as additional text field)
    function addMathModeToHandle(h)
        if isempty(strfind(h.String,'$'))
            h.String=['$',h.String,'$'];
        end
    end
    if useLaTeX
        if useMathXAxis,addMathModeToHandle(allText(1));end
        if useMathYAxis,addMathModeToHandle(allText(2));end
        if numel(allText)>2
            if useMathZAxis,addMathModeToHandle(allText(3));end
        end
        tickLabels={allAxes(1).XTickLabel,allAxes(1).YTickLabel,allAxes(1).ZTickLabel};
        for i=1:numel(tickLabels)
            for j=1:numel(tickLabels{i})
                % If is number of includes pi, add $ (only if not yet
                % included)
                if (~isnan(str2double(tickLabels{i}{j}))||contains(tickLabels{i}{j},'\pi'))&&~contains(tickLabels{i}{j},'$')
                    tickLabels{i}{j}=['$',tickLabels{i}{j},'$'];
                end
            end
        end
        allAxes(1).XTickLabel=tickLabels{1};allAxes(1).YTickLabel=tickLabels{2};allAxes(1).ZTickLabel=tickLabels{3};
        set(allAxes,'ticklabelinterpreter','latex');
        set(findobj(gcf,'type','legend'),'Interpreter','latex');
        set(allText(isvalid(allText)),'interpreter','latex');
    else
        fontName='Times';
        set(findall(gcf, 'Type', 'Axes'), 'FontName', fontName);
        set(allText(isvalid(allText)), 'FontName', fontName);
    end
    set(gcf, ...
                ...%'Color', 'None', ... use -transparent instead when calling export_fig
                'InvertHardCopy', 'Off', ...
                'Units', 'centimeters', ...
                'Position', [0 0 paperSize], ...
                'PaperPositionMode', 'manual', ...
                'PaperUnits', 'centimeters', ...
                'PaperSize', paperSize, ...
                'PaperPosition', [0 0 paperSize]);
    
    
end