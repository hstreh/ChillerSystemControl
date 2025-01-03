
clear all;
clc;
close all;

% read data and change collumn names
data = readtable('ABSData3.csv');


newColumnNames = {'Time', 'deltaP_78', 'deltaP_14', 'deltaP_12', 'Tfw', 'ffw', ...
                   'vfw_1', 'vfw_2', 'vfw_3', 'Chiller_2', 'Chiller_1', ...
                   'Psw', 'Tsw_in1', 'Tsw_in2', 'Tsw_in3' ...
                   'vsw_1', 'vsw_2', 'C_1', 'C_3', 'C_5', 'C_7' ...
                   'C_4', 'C_2', 'C_6', 'C_8', 'P1_suction', 'P2_suction' ...
                   'P1_discharge', 'P2_discharge', 'Tsw_chiller1','Tsw_chiller2' ...
                   'Tout_chiller1','Tout_chiller2', 'T_chiller1','T_chiller2', 'Tfw_sp', 'Tfw_sp2' ...
                   'Tsw_PC', 'Tfw_PCin','Tfw_PCout','Pfw_return','Valve_chiller1', 'Pfw','Valve_chiller2', 'Valve_sw1','Valve_sw2'};

data.Properties.VariableNames = newColumnNames;

%% Ensure the column is a cell array of strings % Convert 'Inactive' to 0 and 'Active' to 1

columnsToConvert = {'Chiller_1', 'Chiller_2', 'Valve_chiller1', 'Valve_chiller2'};
for i = 1:length(columnsToConvert)
    colName = columnsToConvert{i};
    % Convert 'Inactive' to 0 and 'Active' to 1
    if iscell(data.(colName))
        data.(colName) = cellfun(@(x) strcmp(x, 'Active'), data.(colName));
        % Convert logical array to numeric array: true (1) for Active, false (0) for Inactive
    else
        error('Column %s is not a cell array of strings.', colName);
    end
end
%% % Loop over the columns and check for '<null>' in each to remove data

% % Loop over the columns and check for '<null>' in each to remove data

rowsToRemove = false(height(data), 1);

% Iterate over columns 1 to 44 only
for col = 1:44
    if iscellstr(data{:, col}) || isstring(data{:, col}) || iscategorical(data{:, col})
        rowsToRemove = rowsToRemove | strcmp(data{:, col}, '<null>');
    end
end

% Remove rows marked for removal
dataClean = data(~rowsToRemove, :);

% Restrict NaN removal to first 44 columns
nanCheckCols = 1:44;
nanRowsToRemove = any(ismissing(dataClean(:, nanCheckCols)), 2);
dataClean = dataClean(~nanRowsToRemove, :);
%% 

if width(dataClean) >= 46
    % Replace NaN values in column 46 with 0
    dataClean{isnan(dataClean{:, 46}), 46} = 0;
end


%% Divide into seawater cooling and chiller unit cooling

chiller = {'Chiller_1', 'Chiller_2'};

allCompCols = [chiller];

chillerData = dataClean{:, allCompCols};

rowsToRemove2 = all(chillerData == 0, 2);

ChillerUnitData = dataClean(~rowsToRemove2, :);
seaWaterdata = dataClean(rowsToRemove2, :);


%% 
% Define the columns to check for 0 values in Chiller_1 and Chiller_2
chiller = {'Chiller_1', 'Chiller_2'};
chillerData = dataClean{:, chiller};

% Get the necessary temperature columns
Tfw = dataClean.Tfw;
Tfw_PCout = dataClean.Tfw_PCout;
Tout_chiller1 = dataClean.Tout_chiller1;
Tout_chiller2 = dataClean.Tout_chiller2;
Tsw_PC = dataClean.Tsw_PC;
Tfw_sp = dataClean.Tfw_sp;

% Find rows where both Chiller_1 and Chiller_2 are 0
chillersInactive = all(chillerData == 0, 2);  % Logical array for both chillers being 0

% Find rows where Tfw_PCout is closer to Tfw than to Tout_chiller1 and Tout_chiller2
closerToTfw = abs(Tfw_PCout - Tfw) < abs(Tfw_PCout - Tout_chiller1) & ...
              abs(Tfw_PCout - Tfw) < abs(Tfw_PCout - Tout_chiller2);

% Find rows where Tsw_PC is at least 2 degrees below Tfw_sp
TswCondition = (Tsw_PC <= (Tfw_sp));

% Combine initial conditions
initialRowsToRemove = chillersInactive & closerToTfw & TswCondition;

% Subset the data by removing the identified rows
ChillerUnitData2 = dataClean(~initialRowsToRemove, :);  % Keep rows that don't match the removal criteria
seaWaterdata2 = dataClean(initialRowsToRemove, :);
%% Transfer rows back to seaWaterdata2 if Valve_sw2 > 0
Valve_sw2 = ChillerUnitData2.Valve_sw2;

% Find rows where Valve_sw2 > 0
rowsToTransferBack = (Valve_sw2 > 0);

% Extract rows to transfer back
rowsToMove = ChillerUnitData2(rowsToTransferBack, :);

% Remove these rows from ChillerUnitData2
ChillerUnitData2 = ChillerUnitData2(~rowsToTransferBack, :);

% Append the rows back to seaWaterdata2
seaWaterdata2 = [seaWaterdata2; rowsToMove];

% Sort seaWaterdata2 to maintain the correct chronological order
seaWaterdata2 = sortrows(seaWaterdata2, 'Time'); % Replace 'TimeColumn' with the actual column name used for sorting


%% Delet data not used in Chiller


% columns to modify
columnsToModifyChiller1 = {'P1_suction', 'Tsw_chiller1', 'P1_discharge', 'Tout_chiller1','T_chiller1','Tsw_chiller1'};
columnsToModifyChiller2 = {'P2_suction', 'Tsw_chiller2', 'P2_discharge', 'Tout_chiller2','T_chiller2','Tsw_chiller2'};


columnsToCheckChiller1 = {'C_1', 'C_3', 'C_5', 'C_7'};
columnsToCheckChiller2 = {'C_2', 'C_4', 'C_6', 'C_8'};

% Find the rows where any of c1, c3, c5, or c7 is below 5.21
rowsToModifyChiller1 = find(all(ChillerUnitData2{:, columnsToCheckChiller1} <= 5.22, 2));
rowsToModifyChiller2 = find(all(ChillerUnitData2{:, columnsToCheckChiller2} <= 5.22, 2));

% Include the next 3 samples after each row where the condition is met
rowsToModifyExtendedChiller1 = [];
for i = 1:length(rowsToModifyChiller1)
    rowsToModifyExtendedChiller1 = [rowsToModifyExtendedChiller1, rowsToModifyChiller1(i):min(rowsToModifyChiller1(i), height(ChillerUnitData2))];
end
rowsToModifyExtendedChiller1 = unique(rowsToModifyExtendedChiller1); % Ensure no duplicates

% Set the specified columns to NaN for the identified rows
for i = 1:length(columnsToModifyChiller1)
    ChillerUnitData2{rowsToModifyExtendedChiller1, columnsToModifyChiller1{i}} = 0;
end


rowsToModifyExtendedChiller2 = [];
for i = 1:length(rowsToModifyChiller2)
    rowsToModifyExtendedChiller2 = [rowsToModifyExtendedChiller2, rowsToModifyChiller2(i):min(rowsToModifyChiller2(i), height(ChillerUnitData2))];
end
rowsToModifyExtendedChiller2 = unique(rowsToModifyExtendedChiller2); % Ensure no duplicates

% Set the specified columns to NaN for the identified rows
for i = 1:length(columnsToModifyChiller2)
    ChillerUnitData2{rowsToModifyExtendedChiller2, columnsToModifyChiller2{i}} = 0;
end
%% 

% Assuming 'seawaterdata' is your table containing the 'Time' column

% Convert 'Time' to datetime if it's not already
seaWaterdata2.Time = datetime(seaWaterdata2.Time, 'InputFormat', 'dd/MM/yyyy HH:mm:ss.SSS');  % Adjust the format if needed

% Calculate the time differences between consecutive samples (in seconds)
timeDifferences = diff(seaWaterdata2.Time);  % Time difference in datetime format

% Convert the time differences to seconds
intervals = seconds(timeDifferences);

% Initialize a new column for the modified intervals
modifiedIntervals = zeros(height(seaWaterdata2), 1);

% Set the first interval to be zero (for the first data point)
modifiedIntervals(1) = 0;

% Loop through the intervals and apply the rule
for i = 2:height(seaWaterdata2)
    if intervals(i-1) > 1296000  % 15 days in sec
        modifiedIntervals(i) = 43200;  % Change the interval to 30 seconds
    else
        modifiedIntervals(i) = intervals(i-1);  % Use the actual time difference
    end
end

% Add the modified intervals as a new column in the table
seaWaterdata2.interval = modifiedIntervals;
%% 
% Assuming 'seawaterdata' now has the 'interval' column
% Initialize the elapsed time column
elapsedTime = zeros(height(seaWaterdata2), 1);

% Set the first row elapsed time to 0 (starting point)
elapsedTime(1) = 0;

% Loop through the rows and accumulate the intervals to create the elapsed time
for i = 2:height(seaWaterdata2)
    elapsedTime(i) = elapsedTime(i-1) + seaWaterdata2.interval(i);
end

% Add the 'elapsed_time' column to the table
seaWaterdata2.elapsed_time = elapsedTime;


%% 

columsForPC = seaWaterdata2(:, {'Time', 'deltaP_78', 'deltaP_14', 'deltaP_12', 'Tfw', 'ffw', ...
                   'vfw_1', 'vfw_2', 'vfw_3', 'Psw', 'Tsw_in1', 'Tsw_in2', 'Tsw_in3', ...
                   'vsw_1', 'vsw_2', 'Tsw_PC', 'Tfw_PCin','Tfw_PCout','Pfw_return', 'Pfw','Valve_sw2', 'elapsed_time','interval'}); 
filename1 = 'PCdata.csv';
writetable(columsForPC, filename1);
fprintf('Data saved to %s\n', filename1);

columsForChiller = ChillerUnitData2(:, {'Time','deltaP_78', 'deltaP_14', 'deltaP_12', 'Tfw', 'ffw', ...
                 'vfw_1', 'vfw_2', 'vfw_3', 'Chiller_1', 'Chiller_2', ...
                 'Psw', 'Tsw_in1', 'Tsw_in2', 'Tsw_in3', ...
                 'vsw_1', 'vsw_2', 'C_1', 'C_3', 'C_5', 'C_7', ...
                 'C_2', 'C_4', 'C_6', 'C_8', 'P1_suction', 'P2_suction', ...
                 'P1_discharge', 'P2_discharge', 'Tsw_chiller1', 'Tsw_chiller2', ...
                 'Tout_chiller1', 'Tout_chiller2', 'T_chiller1', 'T_chiller2', ...
                 'Tfw_sp', 'Pfw_return', 'Pfw'}); 

filename2 = 'Chillerdata.csv';
writetable(columsForChiller, filename2);
fprintf('Data saved to %s\n', filename2);



data = readtable('PCdata.csv');

summary(seaWaterdata2);

%% 


% Sample data (timestamps as strings)
timestamps = seaWaterdata2.Time;

% Convert strings to datetime objects
timestamps_dt = datetime(timestamps, 'InputFormat', 'dd/MM/yyyy HH:mm:ss.SSS');

% Calculate the time differences between consecutive timestamps
time_diffs = diff(timestamps_dt);

% Calculate the mean time difference
mean_time_diff = mean(time_diffs);

% Display the mean time difference in seconds
disp(['Mean sampling time (in seconds): ', num2str(seconds(mean_time_diff))]);
%% 

columsForTime = seaWaterdata2(:, {'elapsed_time','interval'}); 
filename4 = 'TimeData.csv';
writetable(columsForTime, filename4);
fprintf('Data saved to %s\n', filename4);

