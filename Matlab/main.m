
function main()

%% Enter your name
    close all; clear all; % a bit of cleaning
    import .*
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
    
    
    
    
    %gyroscope data
    [left_foot_gyro,right_foot_gyro] = alignGyroscopeTF2AF(data);
    
    subplot(2,1,1);
    plot(left_foot_gyro)
    title('left foot gyro data')
    ylabel('angular velocity in deg/s') 
    xlabel('time samples (sampled at 500Hz)') 
    legend({'transverse plane (yaw)', 'frontal plane (roll)', 'saggital plane (pitch)'},'Location','southwest')

    subplot(2,1,2); 
    plot(right_foot_gyro)
    title('right foot gyro data')
    ylabel('angular velocity in deg/s') 
    xlabel('time samples (sampled at 500Hz)') 
    legend({'transverse plane (yaw)', 'frontal plane (roll)', 'saggital plane (pitch)'},'Location','southwest')
    
    %white background for nicer reports
    set(gcf,'color','w');
    
    %%
    
    %Not used in the report
    
    freq = 1/data.imu.left.time(2);
    data_1 = applyLowpassFilter(left_foot_gyro(:,1),5,freq);
    data_2 = applyLowpassFilter(left_foot_gyro(:,2),5,freq);
    data_3 = applyLowpassFilter(left_foot_gyro(:,3),5,freq);
    
    subplot(3,1,1);
    plot(data_1)
    title('left transverse plane gyro, filtered at 5Hz')
    
    subplot(3,1,2);
    plot(data_2)
    title('left frontal plane gyro, filtered at 5Hz')
    
    subplot(3,1,3);
    plot(data_3)
    title('left saggital plane gyro, filtered at 5Hz')
    
    %%
    
    %Not used in the report
    
    %gyroscope data
    [left_foot_gyro,right_foot_gyro] = alignGyroscopeTF2AF(data);
    
    subplot(2,1,1);
    plot(applyLowpassFilter(left_foot_gyro(:,3),5,freq))
    title('left foot pitch gyro data, filtered at 5Hz')

    subplot(2,1,2); 
    plot(applyLowpassFilter(right_foot_gyro(:,3),5,freq))
    title('right foot pitch gyro data, filtered at 5Hz')
    
    
    
%% Exercice 1.A.2 (Filter)
    % Use the function applyLowpassFilter(InputSignal, CutoffFrequency, SamplingFrequency)
    % to apply the filter on the accelerometer signal 
    % Plot the three signals on a same axis (you can use hold on command)
    % <<< ENTER YOUR CODE HERE >>>
    
    left_foot_acc = data.imu.left.accel;
    right_foot_acc = data.imu.right.accel;
    
    %% unfiltered plot
    %Not used in the report
    subplot(2,1,1);
    
    plot(left_foot_acc(1:2000,:))
    legend({'X', 'Y', 'Z'},'Location','northwest')

    subplot(2,1,2); 
    plot(right_foot_acc(1:2000,:))
    legend({'X', 'Y', 'Z'},'Location','northwest')
    
    set(gcf,'color','w');
    
    %% 12Hz lp plot
    %Not used in the report
    freq = 1/data.imu.left.time(2);
    
    left_foot_acc_12 = applyLowpassFilter(left_foot_acc,12, freq);
    right_foot_acc_12 = applyLowpassFilter(right_foot_acc,12, freq);
    
    subplot(2,1,1);
    plot(left_foot_acc_12)

    subplot(2,1,2); 
    plot(right_foot_acc_12)
    
    set(gcf,'color','w');
    
    %% 5Hz lp plot
    %Not used in the report
    freq = 1/data.imu.left.time(2);
    
    left_foot_acc_5 = applyLowpassFilter(left_foot_acc,5, freq);
    right_foot_acc_5 = applyLowpassFilter(right_foot_acc,5, freq);
    
    subplot(2,1,1);
    plot(left_foot_acc_5)

    subplot(2,1,2); 
    plot(right_foot_acc_5)
    
    set(gcf,'color','w');
    
    %% Report plot with 12Hz and 5Hz cutoffs
    
    freq = 1/data.imu.left.time(2);
    
    left_foot_acc_5 = applyLowpassFilter(left_foot_acc,5, freq);
    right_foot_acc_5 = applyLowpassFilter(right_foot_acc,5, freq);
    
    subplot(3,1,1);
    plot(left_foot_acc(1:1000,1))
    title('left foot accelerometer data, unfiltered')
    ylabel('acceleration in g') 
    xlabel('time samples (sampled at 500Hz)') 
    
    subplot(3,1,2);
    plot(left_foot_acc_12(1:1000,1))
    title('left foot accelerometer data, filtered with a low-pass at 12Hz')
    ylabel('acceleration in g') 
    xlabel('time samples (sampled at 500Hz)') 

    subplot(3,1,3); 
    plot(left_foot_acc_5(1:1000,1))
    title('left foot accelerometer data, filtered with a low-pass at 5Hz')
    ylabel('angular velocity in g') 
    xlabel('time samples (sampled at 500Hz)') 
    
    set(gcf,'color','w');
    
    
%% Exercice 1.A.3 (Event Detection)
    % detect IC, TC, FF for all midswing-to-midswing cycles (Left foot).
    % <<< ENTER YOUR CODE HERE >>>
    
    
    left_midswing = data.imu.left.midswings;
    right_midswing = data.imu.right.midswings;
    
    
    left_foot_acc = data.imu.left.accel;
    right_foot_acc = data.imu.right.accel;
    
    %%
    
    freq = 1/data.imu.left.time(2);
    ff_thresh = -50;
    
    left_windows = cell(size(left_midswing,2)-1,1);
    
    
    left_IC_events = [];
    left_TC_events = [];
    left_FF = [];
    left_init_FF = [];
    left_mid_FF = [];
    
    left_gyro_filtered = applyLowpassFilter(left_foot_gyro(:,3),5,freq);
    
    % segmenting the gyro data midswing to midswing
    
    %left foot
    for i = 2:1:size(left_midswing,2)
       t0 = left_midswing(i-1);
       t1 = left_midswing(i);
       left_windows{i} = left_gyro_filtered(t0:t1);
       left_windows{i} = applyLowpassFilter(left_windows{i},5,freq);
       subplot(6,4,i-1)
       plot(left_windows{i})
       
       %detecting the IC and TC
       half = ceil(size(left_windows{i},1)/2);
       strike_phase = left_windows{i}(1:half);
       lift_phase = left_windows{i}(half+1:size(left_windows{i},1));
       [~,ic_event] = min(strike_phase);
       left_IC_events = [left_IC_events,ic_event];
       [~,tc_event] = min(lift_phase);
       tc_event = tc_event + half;
       left_TC_events = [left_TC_events,tc_event];
       
       %%quantifying FF
       ff_window = left_windows{i}(ic_event:tc_event);
       ff_val = sum(ff_window > ff_thresh);
       [~,initial_FF_win]=max(ff_window > ff_thresh);
       left_FF = [left_FF, ff_val];
       left_init_FF = [left_init_FF,initial_FF_win];
       left_mid_FF = [left_mid_FF,initial_FF_win+fix(ff_val/2)];
       
       %plot the detection
       subplot(6,4,i-1)
       plot(left_windows{i})
       xline(ic_event)
       xline(tc_event)
    end
    
    sgtitle('Pitch gyro data (in deg/s) for midswing-to-midswing segments of left-foot')
    
    %%
    
    freq = 1/data.imu.left.time(2);
    ff_thresh = -50;
    
    right_windows = cell(size(left_midswing,2)-1,1);
    
    right_IC_events = [];
    right_TC_events = [];
    right_FF = [];
    right_init_FF = [];
    right_mid_FF = [];
    
    right_gyro_filtered = applyLowpassFilter(right_foot_gyro(:,3),5,freq);
    
    %right foot
    for i = 2:1:size(right_midswing,2)
       t0 = right_midswing(i-1);
       t1 = right_midswing(i);
       right_windows{i} = right_gyro_filtered(t0:t1);
       %right_windows_flt{i} = applyLowpassFilter(right_windows{i},5,freq);
       
       %detecting the IC and TC
       half = ceil(size(right_windows{i},1)/2);
       strike_phase = right_windows{i}(1:half);
       lift_phase = right_windows{i}(half+1:size(right_windows{i},1));
       [~,ic_event] = min(strike_phase);
       right_IC_events = [right_IC_events,ic_event];
       [~,tc_event] = min(lift_phase);
       tc_event = tc_event+half;
       right_TC_events = [right_TC_events,tc_event];
       
       %%quantifying FF
       ff_window = right_windows{i}(ic_event:tc_event);
       ff_val = sum(ff_window > ff_thresh);
       [~,initial_FF_win]=max(ff_window > ff_thresh);
       right_FF = [right_FF, ff_val];
       right_init_FF = [right_init_FF,initial_FF_win];
       right_mid_FF = [right_mid_FF,initial_FF_win+fix(ff_val/2)];
       
       
       %plot the detection
       subplot(6,4,i-1)
       plot(right_windows{i})
       xline(ic_event)
       xline(tc_event)
    end
    
    sgtitle('Pitch gyro data (in deg/s) for midswing-to-midswing segments of right-foot')
    %%
    
    subplot(2,1,1)
    hist(left_IC_events)
    subplot(2,1,2)
    hist(left_TC_events)
    %%
    
    % add results to the output structure
    scriptOutResults.imu.leftIC = left_IC_events; % insert your detection of IC
    scriptOutResults.imu.leftTC = right_IC_events; % insert your detection of TC
    scriptOutResults.imu.leftFF = left_FF*(1/freq); % insert your detection of FF
    % detect IC, TC, FF for all midswing-to-midswing cycles (Right foot).
    
    % add results to the output structure
    scriptOutResults.imu.rightIC = right_IC_events; % insert your detection of IC
    scriptOutResults.imu.rightTC = right_TC_events; % insert your detection of TC
    scriptOutResults.imu.rightFF = right_FF*(1/freq); % insert your detection of FF
        
%% Exercice 1.A.4 (Plot Events)
    % plot detection results for right or left foot
    
    %plotting 4 cycles
    subplot(1,1,1)
    gyro = right_gyro_filtered(1:right_midswing(5));
    
    
    plot(gyro);
    xlim([650,3000])
    
    
    for t = 1:1:4
        xline(right_midswing(t),'--')
        xline(right_IC_events(t)+right_midswing(t),'-r')
        xline(right_TC_events(t)+right_midswing(t),'-g')
        xline(right_midswing(t)+right_IC_events(t)+right_mid_FF(t),'-m')
    end
    legend({'Filtered Right Foot Gyro Data','Midswing events','IC events','TC events','Middle of FF phase'},'Location','northeast')
    xlabel('time samples (sampled at 500Hz), events denoted by vertical lines') 
    ylabel('right saggital gyro, filtered at 5Hz') 
%% Exercice 1.A.5 (Compute Gait Cycle, Cadence, Stance Percentage)
    % compute the stance phase percentage, gait cycle time and cadence for
    % left and right leg.
    
    right_gait_times = [];
    for i = 2:1:size(right_midswing,2)
       right_gait_times = [right_gait_times, (1/freq)*(right_midswing(i)-right_midswing(i-1))];
    end
        
    left_gait_times = [];
    for i = 2:1:size(right_midswing,2)
       left_gait_times = [left_gait_times, (1/freq)*(right_midswing(i)-right_midswing(i-1))];
    end
    
    
    
    
    scriptOutResults.imu.leftMeanGaitCycleTime = [mean(left_gait_times)]; % insert the mean gait cycle time of the left foot
    scriptOutResults.imu.leftSTDGaitCycleTime = [std(left_gait_times)]; % insert the gait cycle time STD of the left foot
    scriptOutResults.imu.rightMeanGaitCycleTime = [mean(right_gait_times)]; % insert the mean gait cycle time of the right foot
    scriptOutResults.imu.rightSTDGaitCycleTime = [std(right_gait_times)]; % insert the gait cycle time STD of the right foot
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
    %%
    close all 
    
    subplot(2,1,1)
    fft_plot(right_foot_acc(:,3)-mean(right_foot_acc(:,3)),freq);
    title('Fast Fourier Transform of Right Foot Vertical Acceleration')
    
    subplot(2,1,2)    
    fft_plot(right_foot_gyro(:,3)-mean(right_foot_gyro(:,3)),freq);
    title('Fast Fourier Transform of Right Foot Gyroscope')
    
    set(gcf,'color','w');
    
%% Exercice 1.A.10 (Bonus)(Estimate stride from fft)
% Estimate stride time from fft
    % <<< ENTER YOUR CODE HERE  >>>
    
    
%% Exercice 1.B.1 (Estimate the Pitch Angle)
    % compute the pitch angle from the gyroscope pitch angular velocity.
    % <<< ENTER YOUR CODE HERE >>>
    close all
    freq = 1/data.imu.left.time(2);
    
    %%
    
    flt_d = applyLowpassFilter(right_foot_gyro(:,3),5,freq);
    intgr = cumtrapz(flt_d);
    plot(intgr)
    
     
    
%% Exercice 1.B.2 (Remove the Drift)
    %Brouillon code, faudrait faire pareil mais moins moche

    fit = polyfit(1:1:size(intgr),intgr,1);
    mean_est = polyval(fit,1:1:size(intgr));
    
    subplot(2,1,1)
    plot(intgr)
    hold on 
    plot(mean_est)
    hold off
    
    
    averaged = intgr-mean_est';
    subplot(2,1,2)
    plot(averaged)
    
    %%
    corrected_right_angle = [];
    %right foot
    for i = 2:1:size(right_mid_FF,2)
       t0 = right_mid_FF(i-1)+right_midswing(i-1);
       t1 = right_mid_FF(i)+right_midswing(i);
       right_windows{i} = right_foot_gyro(t0:t1,3);
       right_windows_flt{i} = applyLowpassFilter(right_windows{i},5,freq);

       intgr = cumtrapz(right_windows_flt{i});
       
       slope = intgr(end)/size(intgr,1);
       correction = (1:1:size(intgr,1))*slope;
       %fit = polyfit(1:1:size(intgr),intgr,1);
       %mean_est = polyval(fit,1:1:size(intgr));
       corrected = (intgr-correction');
       
       subplot(6,4,i-1)
       plot(corrected)
       
       corrected_right_angle = [corrected_right_angle,corrected'];
    end
    %%
    subplot(1,1,1)
    
    plot(corrected_right_angle(500:6000))
    xlim([0,5500])
    title('corrected integrated foot angle using foot flat detection')
    xlabel('time samples (sampled at 500Hz)')
    ylabel('angle (in degrees)')
    %%
    subplot(1,1,1)
    plot(corrected_right_angle())
    %%
    
    
    drift_sale = [];
    for i = 2:1:size(right_midswing,2)
       drift_sale = [drift_sale, cumtrapz(right_windows{i}')];
    end
    
    close all
    
    subplot(2,1,1)
    plot(drift_sale)
    
    subplot(2,1,2)
    plot(averaged)
    
    %% Drift Unser Edition
    
    nik_les_drifts = movmean(right_foot_gyro(:,3),300);
    plot(nik_les_drifts)
    
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
   
    TFg = mean(data.imu.right.accelstatic, 1)
    
    scriptOutResults.imu.rightGravityTF = avg_static; % insert right foot TFg here
    
%% Exercice 2.A.2 (Gravity vector in the AF)
    % Express the gravity vector in the anatomical frame
    % <<< No CODE >>>
    Y_AF = [0,-1,0]
            
%% Exercice 2.A.3 (Extract the rotation matrix between TFg and Y_AF )
    % find R_TFg_Y_AF between TFg and Y_AF
    % <<< ENTER YOUR CODE HERE >>>
    %Y_AF = data.imu.right.accelstatic(:,2);
    
    R_TFg_Y_AF = get3DRotationMatrixA2B(Y_AF,TFg)
   
        
%% Exercice 2.A.4 (Plot gravity before and after rotation) 
    % plot the static signals before and after the rotation
    % <<< ENTER YOUR CODE HERE >>>
    %rotate the acceleration signal
    right_AF_accelstatic = zeros(size(data.imu.right.accelstatic));
    for i = 1:1:size(data.imu.right.accelstatic,1)
       right_AF_accelstatic(i,:) = data.imu.right.accelstatic(i,:) * R_TFg_Y_AF;
    end
    %%
    subplot(2,1,1)
    plot(data.imu.right.accelstatic,'-r', 'DisplayName','Before rotation')
    
    subplot(2,1,2)
    plot(right_AF_accelstatic,'-b', 'DisplayName', 'After rotation')
    
    set(gcf,'color','w');
    
%% Exercice 2.A.5 (Describe a method for alignment in the transvers plane) 
    % <<< No CODE >>>
           
%% Exercice 2.B.1 (Plot the three components of the leftCenterFoot marker)
    % plot the three components of the leftCenterFoot marker during walking
    % label the direction of walking and vertical component in the plot
    % <<< ENTER YOUR CODE HERE >>>
    
    freq = data.motioncameras.static.fs;
   
    Z = applyLowpassFilter(data.motioncameras.walking.leftCenterFoot(:,1), 10, freq);
    X = applyLowpassFilter(data.motioncameras.walking.leftCenterFoot(:,2), 10, freq);
    Y = applyLowpassFilter(data.motioncameras.walking.leftCenterFoot(:,3), 10, freq);
    
    subplot(3,1,1)
    plot(Z, 'LineWidth',1);
    title('Z axis') % acceleration is approximately constant, amplitude 0.2
    xlim([0,3000])
    
    subplot(3,1,2)
    plot(X, 'LineWidth',1);
    title('X axis') % cycles, with amplitude 0.5
    xlim([0,3000])
    
    subplot(3,1,3)
    plot(Y, 'LineWidth',1);
    title('Y axis') % varies in a similar manner as above, amplitude 0.4
    xlim([0,3000])
        
%% Exercice 2.B.2 (Construct the technical frame of the left foot)
    % construct the technical frame of the left foot
    % <<< ENTER YOUR CODE HERE >>>
    
    center_m = mean(data.motioncameras.static.leftCenterFoot, 1);
    lateral_m = mean(data.motioncameras.static.leftLateralFoot, 1);
    medial_m = mean(data.motioncameras.static.leftMedialFoot, 1);
    
    scriptOutResults.motioncameras.tfX = [center_m(2), lateral_m(2), medial_m(2)]; % insert TF x-axis
    scriptOutResults.motioncameras.tfY = [center_m(3), lateral_m(3), medial_m(3)]; % insert TF y-axis
    scriptOutResults.motioncameras.tfZ = [center_m(1), lateral_m(1), medial_m(1)]; % insert TF z-axis

        
%% Exercice 2.B.3 (Compute the rotation matrix)
    % compute R_TF_GF
    % <<< ENTER YOUR CODE HERE >>>
    TF = [scriptOutResults.motioncameras.tfX; 
          scriptOutResults.motioncameras.tfY; 
          scriptOutResults.motioncameras.tfZ];
      
    GF = [1, 0, 0;
          0, 1, 0;
          0, 0, 1]; % Tout simplement ?
    
    scriptOutResults.motioncameras.R_TF_GF = get3DRotationMatrixA2B(TF,GF); % insert R_TF_GF
        
%% Exercice 2.B.4 (Construct the anatomical frame of the left foot)
    % construct the anatomical frame of the left foot
    % <<< ENTER YOUR CODE HERE >>>
    data.motioncameras.static.leftMedialMalleolus;
    
    scriptOutResults.motioncameras.afX = []; % insert AF x-axis
    scriptOutResults.motioncameras.afY = []; % insert AF y-axis
    scriptOutResults.motioncameras.afZ = []; % insert AF z-axis
        
%% Exercice 2.B.5 (Orthogonality)
    % Check the orthogonality of the defined coordinate system
    % <<< ENTER YOUR CODE HERE >>>
    
    % If orthogonal, <TF_i, TF_j> = 0 for i ? j
    if (scriptOutResults.motioncameras.afX * scriptOutResults.motioncameras.afY == 0 || ...
        scriptOutResults.motioncameras.afX * scriptOutResults.motioncameras.afZ == 0 || ...
        scriptOutResults.motioncameras.afY * scriptOutResults.motioncameras.afZ == 0) 
        
        disp('Careful : The vector basis for TF is not orthogonal')
    end

%% Exercice 2.B.6 (Compute the rotation matrix between AF and GF)
    % compute R_AF_GF
    % <<< ENTER YOUR CODE HERE >>>
    
    AF = [scriptOutResults.motioncameras.afX; 
          scriptOutResults.motioncameras.afY; 
          scriptOutResults.motioncameras.afZ];
      
    GF = [1, 0, 0;
          0, 1, 0;
          0, 0, 1]; % Tout simplement ?
    
    scriptOutResults.motioncameras.R_AF_GF = get3DRotationMatrixA2B(AF,GF); % insert R_AF_GF
      
%% Exercice 2.B.7 (Compute the rotation matrix between TF and AF)
    % compute R_TF_AF
    % <<< ENTER YOUR CODE HERE >>>
    
    scriptOutResults.motioncameras.R_TF_AF = get3DRotationMatrixA2B(TF, AF); % insert R_TF_AF
       
%% Exercice 2.C.1 (compute TF for walking)
    % (1) compute TF for walking
    % <<< ENTER YOUR CODE HERE >>>
    center_TF = mean(data.motioncamerasa.walking.leftCenterFoot, 1);
    Lateral_TF = mean(data.motioncamerasa.walking.leftLateralFoot, 1);
    medial_TF = mean(data.motioncamerasa.walking.leftMedialFoot, 1);
    
    GF = [center_TF(2), lateral_TF(2), medial_TF(2); 
          center_TF(3), lateral_TF(3), medial_TF(3);
          center_TF(1), lateral_TF(1), medial_TF(1)];
      
    TF = inv(scriptOutResults.motioncameras.R_TF_GF) * GF;
    
%% Exercice 2.C.2 (compute AF for walking)
    % (2) compute AF for walking
    % <<< ENTER YOUR CODE HERE >>>
    AF = inv(scriptOutResults.motioncameras.R_TF_GF) * GF;
    
%% Exercice 2.C.3 (compute the pitch angle)    
    % (3) compute the pitch angle
    % <<< ENTER YOUR CODE HERE >>>
    y = [0, 1, 0]; % Vector orthogonal to the Z0-X0 plane (y-axis).
    n = []; % à définir
    
    alpha = asin(abs(y * n)/abs(y)*abs(n));
 
%% Exercice 2.C.4 (Plot pitch angle and show swing, stance phase, flat foot periods)    
    % (3) compute the pitch angle
    % <<< ENTER YOUR CODE HERE >>>
    alpha_filtered = applyLowpassFilter(alpha, 5, freq); % définir mieux le threshold
    
    plot(t, alpha_filtered, '-b')
    xlabel('time [s]')
    ylabel('Pitch Angle [rad]')
    legend()
          
%%  ------------- (3) ASSIGNEMENT 3: KINETIC ANALYSIS -----------------   

%% Exercice 3.A.1 (extract events using insole)
    % compute the force signal for all cell (transform pressure into force)
    % <<< ENTER YOUR CODE HERE >>>
    
    F_rear_right = (data.insole.right.pressure(:, 1:33)).*(data.insole.right.area(:, 1:33));
    F_fore_right = (data.insole.right.pressure(:, 55:99)).*(data.insole.right.area(:, 55:99));
    
    freq = data.insole.fs;
    
    filtered_F_rear_right = applyLowpassFilter(F_rear_right,5,freq);
    filtered_F_fore_right = applyLowpassFilter(F_fore_right,5,freq);
    
    mean_F_rear_right = sum(filtered_F_rear_right, 2);
    mean_F_fore_right = sum(filtered_F_fore_right, 2);
       
    % detect the sample index of the multiple IC, TS, HO, TO
    % <<< ENTER YOUR CODE HERE >>>
    weight = 70.0;
    threshold = 0.05 * weight;
    
    % store IC, TS, HO and TO detection index
    scriptOutResults.insole.rightHS = find(mean_F_rear_right > threshold); % insert the index of the right foot IC events
    scriptOutResults.insole.rightTS = find(mean_F_fore_right > threshold); % insert the index of the right foot TS events
    scriptOutResults.insole.rightHO = find(mean_F_rear_right < threshold); % insert the index of the right foot HO events
    scriptOutResults.insole.rightTO = find(mean_F_fore_right < threshold); % insert the index of the right foot TO events
   
%% Exercice 3.A.2 (Plot F-rear and F_forefoot)
    % plot a graph showing F_rear and F_Fore at least two
    % strides of the right foot where HS, TS, HO and TO events are 
    % correctly detected and show these event in your plot. Do not forget 
    % to add labels on each axis and a legend for all signals. 
    % <<< ENTER YOUR CODE HERE >>>
    
    x_rear = 1:1:size(mean_F_rear_right,1);
    x_fore = 1:1:size(mean_F_fore_right,1);
    
    subplot(2,1,1)
    plot(mean_F_rear_right(1:400), 'black')
    ylabel('Force [kPa / mm^2)]')
    title('Rear Force right')
    hold on
    plot(x_rear(scriptOutResults.insole.rightHS), mean_F_rear_right(scriptOutResults.insole.rightHS), '-b', 'DisplayName', 'Heel-Strike')
    plot(x_rear(scriptOutResults.insole.rightHO), mean_F_rear_right(scriptOutResults.insole.rightHO), '-r', 'DisplayName', 'Heel-Off')
    xlim([1 400])
    legend()
    hold off
    
    subplot(2,1,2)
    plot(mean_F_fore_right(1:400), 'black')
    ylabel('Force [kPa / mm^2]')
    title('Fore Force right')
    hold on
    plot(x_fore(scriptOutResults.insole.rightTS), mean_F_fore_right(scriptOutResults.insole.rightTS), '-b', 'DisplayName', 'Toe-Strike')
    plot(x_fore(scriptOutResults.insole.rightTO), mean_F_fore_right(scriptOutResults.insole.rightTO), '-r', 'DisplayName', 'Toe-Off')
    xlim([1 400])
    legend()
    hold off
        
%% Exercice 3.A.3 (estimate the foot-flat duration)
    % for the two cycles above, estimate the foot-flat duration
    % <<< ENTER YOUR CODE HERE >>>
        
%% Exercice 3.B.1 (Mean vertical force during flat foot)
    % estimate the total vertical force signal recorded by the insole 
    % during one foot-flat period.
    % <<< ENTER YOUR CODE HERE >>>
    
    id_HO = scriptOutResults.insole.rightHO(1);
    id_TS = scriptOutResults.insole.rightTS(1);
    
    force_cycle = data.insole.right.pressure(id_HO:id_TS,:).*data.insole.right.area(id_HO:id_TS,:);
    
    freq = data.insole.right.fs;
    filtered_force_cycle = applyLowpassFilter(force_cycle,5,freq);
    
    total_force = sum(filtered_force_cycle, 2)

%% Exercice 3.B.2 (free body diagram)
    % <<< No CODE  >>>
    FF_time = mean(data.insole.time(scriptOutResults.insole.rightHO) - data.insole.time(scriptOutResults.insole.rightTS));

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

% I moved the functions to separate files, I thought it was cleaner this way
