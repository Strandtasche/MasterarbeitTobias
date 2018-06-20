
function trackHistoryUpdated = preprocessTrackHistory(ydescending, varargin)
%testSuite: check if
%   Detailed explanation goes here
    assert(ismember(nargin, [1, 2, 3, 4]), 'Wrong number of arguments')
    
    %check if first argument is filename or object and load accordingly
    if isa(varargin{1}, 'char') || isstring(varargin{1})
        load(varargin{1});
    else
        trackHistory_dummy = varargin{1};
    end
    
    %trackHistoryUpdated = 0;
    
    if nargin == 1
        threshold_minNans = 5;
        threshold_backwards = 3;
    elseif nargin == 2
        threshold_minNans = varargin{2};
        threshold_backwards = 3;
    else
        threshold_minNans = varargin{2};
        threshold_backwards = varargin{3};
    end
    
    ok = 1;
    %Removal reverse order
    removal = [];
    %iterate over all Tracks in Trackhistory
    for it = 1:size(trackHistory_dummy,2)
        %calculate number of non-NaN
        nanified = isnan(transpose(trackHistory_dummy(it).RawMeasurements));
        invertedNani = sum(arrayfun(@(x) 1-x, nanified));
        %disp(invertedNani(1))
        if invertedNani(1) < threshold_minNans
            %disp(strcat("not enough values in track no. ", num2str(it)))
            ok = 0;
            removal = [it, removal];
        end
    end

    %removal of tracks with too many NaN Values
    for rem = removal
        trackHistory_dummy(rem) = [];
    end
    
    %now check in each track if the observations are going "backwards"
    backward_tracks = [];
    for it=1:size(trackHistory_dummy,2)
        count = 0;
        lastObservation = [0 0];
        currentTrackRawMeasurements = trackHistory_dummy(it).RawMeasurements';
        for obs_it=1:size(currentTrackRawMeasurements,1)
            newObservation = currentTrackRawMeasurements(obs_it,:);
            if ydescending
                if newObservation(2) < lastObservation(2)
                    count = count + 1;
                end
            else
                if newObservation(2) > lastObservation(2)
                    count = count + 1;
                end
            end
            
            lastObservation = newObservation;
        end
        if count >= threshold_backwards 
            backward_tracks(:, end+1) = [it it];
        end
    end
    
    if ok == 1
        disp("All correct!")
    end
    trackHistory_dummy(backward_tracks) = [];
    
    trackHistoryUpdated = trackHistory_dummy;
    %disp(trackHistoryUpdated)
end


%disp(trackHistory(1).RawMeasurements)


