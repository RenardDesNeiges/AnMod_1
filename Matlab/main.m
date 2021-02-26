
function main()

%% Enter your name
    close all; clear all; % a bit of cleaning
    %======================================================================
    % 1)  ASSIGNEMENT 0: SETTINGS
    %======================================================================
    % replace by your own information
    LAST_NAME1 = 'Renard'; % Last name 1
    LAST_NAME2 = 'Girard'; % Last name 2
    DATASET_NUMBER = 29; % your dataset number
    
    % settings
    DATASET_NAME = ['amlWalkingDataStruct' num2str(DATASET_NUMBER) '.mat'];
    load(DATASET_NAME); % load data structure
    scriptOutResults = [];
      
%%  ------------- (1) ASSIGNEMENT 1: TEMPORAL AND SPATIAL GAIT ANALYSIS -------------------    

%% Exercice 1.A.1 (Pitch_Angle Rotate to Align with Anatomical Frame)

    % align gyroscope TF with the foot AF using function
    % alignGyroscope(data) and plot the results
    % <<< ENTER YOUR CODE HERE >>>
    
%% Exercice 1.A.2 (Filter)
    % Use the function applyLowpassFilter(InputSignal, CutoffFrequency, SamplingFrequency)
    % to apply the filter on the accelerometer signal 
    % Plot the three signals on a same axis (you can use hold on command)
    % <<< ENTER YOUR CODE HERE >>>
    
%% Exercice 1.A.3 (Event Detection)
    % detect IC, TC, FF for all midswing-to-midswing cycles (Left foot).
    % <<< ENTER YOUR CODE HERE >>>
    
    % add results to the output structure
    scriptOutResults.imu.leftIC = []; % insert your detection of IC
    scriptOutResults.imu.leftTC = []; % insert your detection of TC
    scriptOutResults.imu.leftFF = []; % insert your detection of FF
    % detect IC, TC, FF for all midswing-to-midswing cycles (Right foot).
    % <<< ENTER YOUR CODE HERE >>>
    
    % add results to the output structure
    scriptOutResults.imu.rightIC = []; % insert your detection of IC
    scriptOutResults.imu.rightTC = []; % insert your detection of TC
    scriptOutResults.imu.rightFF = []; % insert your detection of FF
        
%% Exercice 1.A.4 (Plot Events)
    % plot detection results for right or left foot
    % <<< ENTER YOUR CODE HERE >>>
    
%% Exercice 1.A.5 (Compute Gait Cycle, Cadence, Stance Percentage)
    % compute the stance phase percentage, gait cycle time and cadence for
    % left and right leg.
    % <<< ENTER YOUR CODE HERE >>>
    
    scriptOutResults.imu.leftMeanGaitCycleTime = []; % insert the mean gait cycle time of the left foot
    scriptOutResults.imu.leftSTDGaitCycleTime = []; % insert the gait cycle time STD of the left foot
    scriptOutResults.imu.rightMeanGaitCycleTime = []; % insert the mean gait cycle time of the right foot
    scriptOutResults.imu.rightSTDGaitCycleTime = []; % insert the gait cycle time STD of the right foot
    scriptOutResults.imu.leftMeanCadence = []; % insert the mean cadence of the left foot
    scriptOutResults.imu.leftSTDCadence = []; % insert the cadence STD of the left foot
    scriptOutResults.imu.rightMeanCadence = []; % insert the mean cadence of the left foot
    scriptOutResults.imu.rightSTDCadence = []; % insert the cadence STD of the left foot
    scriptOutResults.imu.leftMeanStance = []; % insert the mean stance phase duration of left foot
    scriptOutResults.imu.leftSTDStance = []; % insert the stance phase duration STD of left foot
    scriptOutResults.imu.rightMeanStance = []; % insert the mean stance phase duration of right foot
    scriptOutResults.imu.rightSTDStance = []; % insert the stance phase duration STD of right foot

%% Exercice 1.A.6 (Compare the mean right/left cadence)
    % Compare the mean cadence of the right leg to the right leg 
    % <<< No CODE >>>

%% Exercice 1.A.7 (Estimate the coefficient of variation GC_Time)
    % Estimate the coefficient of variation (in %) of the gait cycle time 
    % obtained from of the right foot
    % <<< ENTER YOUR CODE HERE >>>

    
    scriptOutResults.imu.cvGCT = []; % insert CV GCT right foot

%% Exercice 1.A.8 (Propose another method to extract Cadence)
    % <<< No CODE >>>
  
%% Exercice 1.A.9 (Fast fourier transform)
% You can use fft_plot function
    % <<< ENTER YOUR CODE HERE  >>>
    
%% Exercice 1.A.10 (Bonus)(Estimate stride from fft)
% Estimate stride time from fft
    % <<< ENTER YOUR CODE HERE  >>>
    
    
%% Exercice 1.B.1 (Estimate the Pitch Angle)
    % compute the pitch angle from the gyroscope pitch angular velocity.
    % <<< ENTER YOUR CODE HERE >>>
      
%% Exercice 1.B.2 (Remove the Drift)
    
    % correct the drift on the pitch angle signal
    % <<< ENTER YOUR CODE HERE >>>
       
%% Exercice 1.B.3 (Plot the Drift_free and Drifted Pitch Angle)(Bonus)
    % plot gyroscope pitch angular velocity, pitch angle with and without
    % drift
    % <<< ENTER YOUR CODE HERE >>>

%% Exercice 1.B.4 (Mean and STD of Pitch angle at IC)(Bonus)
% <<< ENTER YOUR CODE HERE >>>

%% ------------- (2) ASSIGNEMENT 2: FRAME & ORIENTATIONS -------------------    

%% Exercice 2.A.1 (Gravity vector in the right foot IMU TF)
    % gravity vector in the right foot IMU TF
    % <<< ENTER YOUR CODE HERE >>>
    
    scriptOutResults.imu.rightGravityTF = []; % insert right foot TFg here
    
%% Exercice 2.A.2 (Gravity vector in the AF)
    % Express the gravity vector in the anatomical frame
    % <<< No CODE >>>
        
%% Exercice 2.A.3 (Extract the rotation matrix between TFg and Y_AF )
    % find R_TFg_Y_AF between TFg and Y_AF
    % <<< ENTER YOUR CODE HERE >>>
    
    scriptOutResults.imu.rightRotationYAF = []; % insert R_TFg_Y_AF
        
%% Exercice 2.A.4 (Plot gravity before and after rotation) 
    % plot the static signals before and after the rotation
    % <<< ENTER YOUR CODE HERE >>>
    
%% Exercice 2.A.5 (Describe a method for alignment in the transvers plane) 
    % <<< No CODE >>>
           
%% Exercice 2.B.1 (Plot the three components of the leftCenterFoot marker)
    % plot the three components of the leftCenterFoot marker during walking
    % label the direction of walking and vertical component in the plot
    % <<< ENTER YOUR CODE HERE >>>
        
%% Exercice 2.B.2 (Construct the technical frame of the left foot)
    % construct the technical frame of the left foot
    % <<< ENTER YOUR CODE HERE >>>
   
    
    
    scriptOutResults.motioncameras.tfX = []; % insert TF x-axis
    scriptOutResults.motioncameras.tfY = []; % insert TF y-axis
    scriptOutResults.motioncameras.tfZ = []; % insert TF z-axis
        
%% Exercice 2.B.3 (Compute the rotation matrix)
    % compute R_TF_GF
    % <<< ENTER YOUR CODE HERE >>>
    
    scriptOutResults.motioncameras.R_TF_GF = []; % insert R_TF_GF
        
%% Exercice 2.B.4 (Construct the anatomical frame of the left foot)
    % construct the anatomical frame of the left foot
    % <<< ENTER YOUR CODE HERE >>>
    
    scriptOutResults.motioncameras.afX = []; % insert AF x-axis
    scriptOutResults.motioncameras.afY = []; % insert AF y-axis
    scriptOutResults.motioncameras.afZ = []; % insert AF z-axis
        
%% Exercice 2.B.5 (Orthogonality)
    % Check the orthogonality of the defined coordinate system
    % <<< ENTER YOUR CODE HERE >>>

%% Exercice 2.B.6 (Compute the rotation matrix between AF and GF)
    % compute R_AF_GF
    % <<< ENTER YOUR CODE HERE >>>    
    
    scriptOutResults.motioncameras.R_AF_GF = []; % insert R_AF_GF
      
%% Exercice 2.B.7 (Compute the rotation matrix between TF and AF)
    % compute R_TF_AF
    % <<< ENTER YOUR CODE HERE >>>
    
    scriptOutResults.motioncameras.R_TF_AF = []; % insert R_TF_AF
       
%% Exercice 2.C.1 (compute TF for walking)
    % (1) compute TF for walking
    % <<< ENTER YOUR CODE HERE >>>
    
%% Exercice 2.C.2 (compute AF for walking)
    % (2) compute AF for walking
    % <<< ENTER YOUR CODE HERE >>>
    
%% Exercice 2.C.3 (compute the pitch angle)    
    % (3) compute the pitch angle
    % <<< ENTER YOUR CODE HERE >>>
 
%% Exercice 2.C.4 (Plot pitch angle and show swing, stance phase, flat foot periods)    
    % (3) compute the pitch angle
    % <<< ENTER YOUR CODE HERE >>>
          
%%  ------------- (3) ASSIGNEMENT 3: KINETIC ANALYSIS -----------------   

%% Exercice 3.A.1 (extract events using insole)
    % compute the force signal for all cell (transform pressure into force)
    % <<< ENTER YOUR CODE HERE >>>
       
    % detect the sample index of the multiple IC, TS, HO, TO
    % <<< ENTER YOUR CODE HERE >>>
    
    % store IC, TS, HO and TO detection index
    scriptOutResults.insole.rightIC = []; % insert the index of the right foot IC events
    scriptOutResults.insole.rightTS = []; % insert the index of the right foot TS events
    scriptOutResults.insole.rightHO = []; % insert the index of the right foot HO events
    scriptOutResults.insole.rightTO = []; % insert the index of the right foot TO events
   
%% Exercice 3.A.2 (Plot F-rear and F_forefoot)
    % plot a graph showing F_rear and F_Fore at least two
    % strides of the right foot where HS, TS, HO and TO events are 
    % correctly detected and show these event in your plot. Do not forget 
    % to add labels on each axis and a legend for all signals. 
    % <<< ENTER YOUR CODE HERE >>>
        
%% Exercice 3.A.3 (estimate the foot-flat duration)
    % for the two cycles above, estimate the foot-flat duration
    % <<< ENTER YOUR CODE HERE >>>
        
%% Exercice 3.B.1 (Mean vertical force during flat foot)
    % estimate the total vertical force signal recorded by the insole 
    % during one foot-flat period.
    % <<< ENTER YOUR CODE HERE >>>

%% Exercice 3.B.2 (free body diagram)
    % <<< No CODE  >>>

%% Exercice 3.B.3 (mean value of ankle net force and moment during foot flat )
    % compute the net force at the ankle (F_A) and the net moment at the
    % ankle (M_A) for every timesample during one footflat period
    % <<< ENTER YOUR CODE HERE >>>
    

    % compute the mean value of F_A and M_A
    % <<< ENTER YOUR CODE HERE >>>
    
     scriptOutResults.insole.MeanF_A = [];
     scriptOutResults.insole.MeanM_A = [];

%% Exercice 3.B.4 (IMU vs. Insole for event detection)
    % compare the IMU with Insole for event detection
    % <<< No CODE >>>

%% Exercice 3.B.5 (GRF )
    % compute the net force apply to the foot during the whole stance phase
    % Plot the GRF for one gait cycle
    % <<< ENTER YOUR CODE HERE >>>
    
%% Save the output   
    %======================================================================
    %  5) ENDING TASKS
    %======================================================================
    % Save the output structure
    save([LAST_NAME1 '_' LAST_NAME2 '_outStruct.mat'],'scriptOutResults');
    
end %function main

%==========================================================================
%   AML LIBRARY
%==========================================================================
function [alignedLeftGyro,alignedRightGyro] = alignGyroscopeTF2AF(data)
% ALIGNGYROSCOPETF2AF aligns the gyroscope TF with the foot AF
%   [A, B] = alignGyroscopeTF2AF(D) returns the angular velocity measured
%   by the gyroscope, but expressed in the anatomical frame of the foot.
%   Here D is the complete data structure given in the project, A is the
%   Nx3 matrix with the left foot angular velocity, and B is the Nx3 matrix
%   with the right foot angular velocity.
    alignedLeftGyro = data.imu.left.gyro * data.imu.left.calibmatrix.';
    alignedRightGyro = data.imu.right.gyro * data.imu.right.calibmatrix.';
end % function

function [R] = get3DRotationMatrixA2B(A,B)
%GET3DROTATIONMATRIXA2B returns the rotation matrix which rotates vector A onto vector B. 
%   This function must be used as such B = R * A with 
%   R = get3DRotationMatrixA2B(A,B).
%
%   INPUTS:
%       - A: vector in 3D space
%       - B: vector in 3D space
%   OUTPUTS:
%       - R: 3x3 rotation matrix

    % The formula used can be found on: 
    % http://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
    A = A(:)/norm(A); B=B(:)/norm(B); % force Nx1 format and normalize
    v = cross(A,B);
    s = norm(v);
    c = dot(A,B);
    Vskew = [0 -v(3) v(2); v(3) 0 -v(1); -v(2) v(1) 0];
    R = eye(3) + Vskew + Vskew^2 * ((1-c)/s^2);
end % function

function Out=applyLowpassFilter(InputSignal,CutoffFrequency, SamplingFrequency)
% This function applies a lowpass butterworth filter to the input signal
% provide the InputSignal, CutoffFrequency and SamplingFrequency as input
    [b,a] = butter(2, 1.2465*CutoffFrequency/SamplingFrequency*2, 'low');
    Out= filtfilt(b,a,InputSignal);
end


function Out= fft_plot (X, Fs)
% Plot the fast fourier transform 
% Input1: X   one dimensional vector  
% Input2: Fs  Sampling frequency        

T = 1/Fs;             % Sampling period       
L = length(X);             % Length of signal
t = (0:L-1)*T;        % Time vector
Y = fft(X);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;
plot(f,P1) 
title('Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')
xlim([0 10])
end