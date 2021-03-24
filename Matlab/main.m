
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
    freq = data.imu.left.fs;
    
    right_gait_times = [];
    for i = 2:1:size(right_midswing,2)
       right_gait_times = [right_gait_times, (1/freq)*(right_midswing(i)-right_midswing(i-1))];
    end
        
    left_gait_times = [];
    for i = 2:1:size(left_midswing,2)
       left_gait_times = [left_gait_times, (1/freq)*(left_midswing(i)-left_midswing(i-1))];
    end
    
    left_cadence = 120 ./ left_gait_times;
    right_cadence = 120 ./ right_gait_times;
    
    left_stance = (left_TC_events - left_IC_events) * (1/ freq);
    right_stance = (right_TC_events - right_IC_events) * (1/ freq);
    

    scriptOutResults.imu.leftMeanGaitCycleTime = [mean(left_gait_times)]; % insert the mean gait cycle time of the left foot
    scriptOutResults.imu.leftSTDGaitCycleTime = [std(left_gait_times)]; % insert the gait cycle time STD of the left foot
    scriptOutResults.imu.rightMeanGaitCycleTime = [mean(right_gait_times)]; % insert the mean gait cycle time of the right foot
    scriptOutResults.imu.rightSTDGaitCycleTime = [std(right_gait_times)]; % insert the gait cycle time STD of the right foot
    scriptOutResults.imu.leftMeanCadence = [mean(left_cadence)]; % insert the mean cadence of the left foot
    scriptOutResults.imu.leftSTDCadence = [std(left_cadence)]; % insert the cadence STD of the left foot
    scriptOutResults.imu.rightMeanCadence = [mean(right_cadence)]; % insert the mean cadence of the left foot
    scriptOutResults.imu.rightSTDCadence = [std(right_cadence)]; % insert the cadence STD of the left foot
    scriptOutResults.imu.leftMeanStance = [mean(left_stance())]; % insert the mean stance phase duration of left foot
    scriptOutResults.imu.leftSTDStance = [std(left_stance)]; % insert the stance phase duration STD of left foot
    scriptOutResults.imu.rightMeanStance = [mean(right_stance)]; % insert the mean stance phase duration of right foot
    scriptOutResults.imu.rightSTDStance = [std(right_stance)]; % insert the stance phase duration STD of right foot

%% Exercice 1.A.6 (Compare the mean right/left cadence)
    % Compare the mean cadence of the right leg to the right leg 
    % <<< No CODE >>>

%% Exercice 1.A.7 (Estimate the coefficient of variation GC_Time)
    % Estimate the coefficient of variation (in %) of the gait cycle time 
    % obtained from of the right foot
    % <<< ENTER YOUR CODE HERE >>>

    
    scriptOutResults.imu.cvGCT = scriptOutResults.imu.rightSTDGaitCycleTime / scriptOutResults.imu.rightMeanGaitCycleTime; % insert CV GCT right foot

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
    
    scriptOutResults.imu.rightGravityTF = TFg; % insert right foot TFg here
    
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
    plot(data.imu.right.accelstatic,'-', 'DisplayName','Before rotation')
    legend({'X', 'Y', 'Z'},'Location','northwest')
    title('static accelerometer measurements in TF')
    xlabel('time samples (sampled at 500Hz)')
    ylabel('measured values (in g)')
    
    subplot(2,1,2)
    plot(right_AF_accelstatic,'-', 'DisplayName', 'After rotation')
    legend({'X', 'Y', 'Z'},'Location','northwest')
    title('static accelerometer measurements in AF')
    xlabel('time samples (sampled at 500Hz)')
    ylabel('measured values (in g)')
    sgtitle('Pitch gyro data (in deg/s) for midswing-to-midswing segments of right-foot')
    set(gcf,'color','w');
    
%% Exercice 2.A.5 (Describe a method for alignment in the transvers plane) 
    % <<< No CODE >>>
           
%% Exercice 2.B.1 (Plot the three components of the leftCenterFoot marker)
    % plot the three components of the leftCenterFoot marker during walking
    % label the direction of walking and vertical component in the plot
    % <<< ENTER YOUR CODE HERE >>>
    
    freq = data.motioncameras.static.fs;
   
    Z = applyLowpassFilter(data.motioncameras.walking.leftCenterFoot(:,1), 10, freq);
    Y = applyLowpassFilter(data.motioncameras.walking.leftCenterFoot(:,2), 10, freq);
    X = applyLowpassFilter(data.motioncameras.walking.leftCenterFoot(:,3), 10, freq);
    
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
    
    set(gcf,'color','w');
    
    sgtitle('Components of the leftCenterFoot marker over time')
        
%% Exercice 2.B.2 (Construct the technical frame of the left foot)
    % construct the technical frame of the left foot
    % <<< ENTER YOUR CODE HERE >>>
    
    % COMMENTS FOR THE CORRECTION
    % we observe that the marker placement doesn't contstruct orthonormal
    % vectors, because we want to form a basis, we assume that the markers
    % are actually in a plane perpendicular to Y_AF (meaning we can infer 
    % the direction of Y_TF with a cross product) and choose arbitrarly 
    % that X_TF is the direction given by tracing a line from marker 1 
    % to marker 2 Z_TF is then computed as a cross product between X_TF 
    % and Y_TF
    
    left_center_m = mean(data.motioncameras.static.leftCenterFoot, 1);
    left_lateral_m = mean(data.motioncameras.static.leftLateralFoot, 1);
    left_medial_m = mean(data.motioncameras.static.leftMedialFoot, 1);
    
    Xtf = (left_lateral_m - left_center_m)/norm(left_lateral_m - left_center_m,2)
    center_medial = (left_medial_m - left_center_m)/norm(left_medial_m - left_center_m,2);
    Ytf = cross(center_medial,Xtf)/norm(cross(center_medial,Xtf),2)
    Ztf = cross(Xtf,Ytf)/norm(cross(Xtf,Ytf),2)
    
    scriptOutResults.motioncameras.tfX = Xtf; % insert TF x-axis
    scriptOutResults.motioncameras.tfY = Ytf; % insert TF y-axis
    scriptOutResults.motioncameras.tfZ = Ztf; % insert TF z-axis
        
%% Exercice 2.B.3 (Compute the rotation matrix)
    % compute R_TF_GF
    
    R1 = get3DRotationMatrixA2B([1,0,0],Xtf);
    R2 = get3DRotationMatrixA2B([0,1,0],Ytf*R1);
    R3 = get3DRotationMatrixA2B([0,0,1],Ztf*R1*R2);

    TF_to_GF = inv(R1 * R2 * R3);
    
    scriptOutResults.motioncameras.R_TF_GF = TF_to_GF; % insert R_TF_GF
    
    %verification :     
    Xgf_check = Xtf * inv(TF_to_GF)
    Ygf_check = Ytf * inv(TF_to_GF)
    Zgf_check = Ztf * inv(TF_to_GF)
        
%% Exercice 2.B.4 (Construct the anatomical frame of the left foot)
    % construct the anatomical frame of the left foot
    % <<< ENTER YOUR CODE HERE >>>
    left_medial = mean(data.motioncameras.static.leftMedialMalleolus,1);
    left_lateral = mean(data.motioncameras.static.leftLateralMalleolus,1);
    
    afZ = (left_medial - left_lateral);
    % here we have a problem : the afZ vector constructed by the markers
    % is not in the plane (it has a non-zero y component) so let's just
    % set it to 0 by hand so that our frame forms a base as expected
    afZ(2) = 0;
    afZ = afZ/norm(afZ,2);
    afY = [0, 1, 0];
    afX = cross(afZ,afY);
    
    
    scriptOutResults.motioncameras.afX = afX; % insert AF x-axis
    scriptOutResults.motioncameras.afY = afY; % insert AF y-axis
    scriptOutResults.motioncameras.afZ = afZ; % insert AF z-axis
    
%% Exercice 2.B.5 (Orthogonality)
    % Check the orthogonality of the defined coordinate system
    % <<< ENTER YOUR CODE HERE >>>
    
    % If orthogonal, <TF_i, TF_j> = 0 for i ? j
    if ((dot(afX,afY)+dot(afX,afZ)+dot(afY,afZ)) ~= 0)
       disp("not orthogonal") 
    end

%% Exercice 2.B.6 (Compute the rotation matrix between AF and GF)
    % compute R_AF_GF
    % <<< ENTER YOUR CODE HERE >>>
    
    R1 = get3DRotationMatrixA2B([1,0,0],afX);   % we only need one rotation 
                                                % since Y axes are aligned
                                                % already
    AF_to_GF = inv(R1*[1,0,0;0,1,0;0,0,-1]);
    
    scriptOutResults.motioncameras.R_AF_GF = AF_to_GF; % insert R_AF_GF
      
    Xgf_check = afX * inv(AF_to_GF)
    Ygf_check = afY * inv(AF_to_GF)
    Zgf_check = afZ * inv(AF_to_GF)
        
%% Exercice 2.B.7 (Compute the rotation matrix between TF and AF)
    % compute R_TF_AF
    % <<< ENTER YOUR CODE HERE >>>
    
    R1 = get3DRotationMatrixA2B(afX,Xtf);
    R2 = get3DRotationMatrixA2B(afY,Ytf*R1);
    R3 = get3DRotationMatrixA2B(afZ,Ztf*R1*R2);

    TF_to_AF = inv(R1 * R2 * R3)
    
    scriptOutResults.motioncameras.R_TF_AF = TF_to_AF; % insert R_TF_AF
       
%% Exercice 2.C.1 (compute TF for walking)
    % (1) compute TF for walking
    % <<< ENTER YOUR CODE HERE >>>
    center_data = data.motioncameras.walking.leftCenterFoot;
    Lateral_data = data.motioncameras.walking.leftLateralFoot;
    medial_data = data.motioncameras.walking.leftMedialFoot;
    
    % COMMENTS 
    
    Xtf_data = (Lateral_data - center_data)./vecnorm(Lateral_data - center_data,2,2);
    medial_data = (medial_data - center_data)./vecnorm(medial_data - center_data,2,2);
    Ytf_data = cross(medial_data,Xtf_data)./vecnorm(cross(medial_data,Xtf_data),2,2);
    Ztf_data = cross(Xtf_data,medial_data)./vecnorm(cross(Xtf_data,medial_data),2,2);
    
    subplot(3,1,1)
    plot(Xtf_data)
    xlim([1,2500])
    subplot(3,1,2)
    plot(Ytf_data)
    xlim([1,2500])
    subplot(3,1,3)
    plot(Ztf_data)
    xlim([1,2500])
    
   
%% Exercice 2.C.2 (compute AF for walking)
    % (2) compute AF for walking
    % <<< ENTER YOUR CODE HERE >>>
    
    Xaf_data =  zeros(size(Xtf_data));
    Yaf_data =  zeros(size(Ytf_data));
    Zaf_data =  zeros(size(Ztf_data));
    
    
    for i = 1:1:size(Xtf_data,1)
        xt = Xtf_data(i,:);
        yt = Ytf_data(i,:);
        zt = Ztf_data(i,:);
        
        xa = xt * inv(TF_to_AF);
        ya = yt * inv(TF_to_AF);
        za = zt * inv(TF_to_AF);
        
        Xaf_data(i,:) = xa;
        Yaf_data(i,:) = ya;
        Zaf_data(i,:) = za;
    end
    
    
    subplot(3,1,1)
    plot(Xaf_data)
    xlim([1,2500])
    subplot(3,1,2)
    plot(Yaf_data)
    xlim([1,2500])
    subplot(3,1,3)
    plot(Zaf_data)
    xlim([1,2500])
    
    %% AF DATA VS TF DATA
    
    subplot(3,2,1)
    plot(Xtf_data)
    title('xtf')
    legend({'X', 'Y', 'Z'},'Location','northwest')
    xlim([1,500])
    subplot(3,2,3)
    plot(Ytf_data)
    title('ytf')
    legend({'X', 'Y', 'Z'},'Location','northwest')
    xlim([1,500])
    subplot(3,2,5)
    plot(Ztf_data)
    title('ztf')
    legend({'X', 'Y', 'Z'},'Location','northwest')
    xlim([1,500])
    
    subplot(3,2,2)
    plot(Xaf_data)
    title('xaf')
    legend({'X', 'Y', 'Z'},'Location','northwest')
    xlim([1,500])
    subplot(3,2,4)
    plot(Yaf_data)
    title('yaf')
    legend({'X', 'Y', 'Z'},'Location','northwest')
    xlim([1,500])
    subplot(3,2,6)
    plot(Zaf_data)
    title('zaf')
    legend({'X', 'Y', 'Z'},'Location','northwest')
    xlim([1,500])
    
%% Exercice 2.C.3 (compute the pitch angle)    
    % (3) compute the pitch angle
    % <<< ENTER YOUR CODE HERE >>>
    % BIG CHUNGUS COMPUTATION
    n = [0, 1, 0]; % Vector orthogonal to the Z0-X0 plane (y-axis).
    
    alpha = zeros(size(Yaf_data,1),1);
    
    for i = 1:1:size(Yaf_data,1)
        
        alpha(i) =  asin(norm(cross(Yaf_data(i,:),n),2) / ...
            (norm(Yaf_data(i,:),2) * norm(n,2)));
    end
    
    %alpha = asin(abs(y * n)/abs(y)*abs(n));
 
%% Exercice 2.C.4 (Plot pitch angle and show swing, stance phase, flat foot periods)    
    % (3) compute the pitch angle
    % <<< ENTER YOUR CODE HERE >>>
    alpha_filtered = applyLowpassFilter(alpha, 5, freq); % définir mieux le threshold
    
    subplot(2,1,1)
    plot(alpha_filtered)
    start = 100 ;
    xlim([start,start+100])
    subplot(2,1,2)
    plot(intgr)
    xlim([100,600])
    
    %plot(t, alpha_filtered, '-b')
    %xlabel('time [s]')
    %ylabel('Pitch Angle [rad]')
    %legend()
          
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
    
    % Heel detection:
    id_sup_rear = find(mean_F_rear_right >= threshold);
    id_inf_rear = find(mean_F_rear_right <= threshold);
    id_inf_rear = [0; id_inf_rear];

    id_sup = id_sup_rear * ones(1, length(id_inf_rear));
    id_inf = id_inf_rear * ones(1, length(id_sup_rear));
    
    id_sup_inf = id_sup - id_inf';
    [i_rear, ~] = find(id_sup_inf == 1);
    [~, j_rear_1] = find(id_sup_inf == -1);
    
    t_HS = id_sup_rear(i_rear);
    t_HO = id_inf_rear(j_rear_1); 
    
    % Toe detection:
    id_sup_fore = find(mean_F_fore_right >= threshold);
    id_inf_fore = find(mean_F_fore_right <= threshold);
    id_inf_fore = [0; id_inf_fore];

    id_sup_f = id_sup_fore * ones(1, length(id_inf_fore));
    id_inf_f = id_inf_fore * ones(1, length(id_sup_fore));
    
    id_sup_inf_f = id_sup_f - id_inf_f';
    [i_fore, ~] = find(id_sup_inf_f == 1);
    [~, j_fore_1] = find(id_sup_inf_f == -1);
    
    t_TS = id_sup_fore(i_fore);
    t_TO = id_inf_fore(j_fore_1); 
    
    % store IC, TS, HO and TO detection index
    scriptOutResults.insole.rightHS = t_HS; % insert the index of the right foot IC events
    scriptOutResults.insole.rightTS = t_TS; % insert the index of the right foot TS events
    scriptOutResults.insole.rightHO = t_HO; % insert the index of the right foot HO events
    scriptOutResults.insole.rightTO = t_TO; % insert the index of the right foot TO events
   
%% Exercice 3.A.2 (Plot F-rear and F_forefoot)
    % plot a graph showing F_rear and F_Fore at least two
    % strides of the right foot where HS, TS, HO and TO events are 
    % correctly detected and show these event in your plot. Do not forget 
    % to add labels on each axis and a legend for all signals. 
    % <<< ENTER YOUR CODE HERE >>>
    
    x_rear = 1:1:size(mean_F_rear_right,1);
    x_fore = 1:1:size(mean_F_fore_right,1);
    
    subplot(2,1,1)
    plot(mean_F_rear_right, 'black')
    ylabel('Force [kPa * mm^2)]')
    title('Rear Force right')
    hold on
    xline(t_HO, '--r', 'Heel-Off')
    xline(t_HS, '--b', 'Heel-Strike')
    xlim([150 400])
    hold off
    
    subplot(2,1,2)
    plot(mean_F_fore_right, 'black')
    ylabel('Force [kPa * mm^2]')
    title('Fore Force right')
    hold on
    xline(t_TO, '--r', 'Toe-Off')
    xline(t_TS, '--b', 'Toe-Strike')
    xlim([150 400])
    hold off
        
%% Exercice 3.A.3 (estimate the foot-flat duration)
    % for the two cycles above, estimate the foot-flat duration
    % <<< ENTER YOUR CODE HERE >>>
    
    % Foot flat duration = t(HO) - t(TS)
    ff_d1 = t_HO(1) - t_TS(1);
    ff_d2 = t_HO(2) - t_TS(2);
        
%% Exercice 3.B.1 (Mean vertical force during flat foot)
    % estimate the total vertical force signal recorded by the insole 
    % during one foot-flat period.
    % <<< ENTER YOUR CODE HERE >>>
    
    force = sum(data.insole.right.pressure.*data.insole.right.area, 2);
    mean_force_FF = mean(force(t_TS(1):t_HO(1)));

%% Exercice 3.B.2 (free body diagram)
    % <<< No CODE  >>>
    FF_time = mean(data.insole.time(scriptOutResults.insole.rightHO) - data.insole.time(scriptOutResults.insole.rightTS));

%% Exercice 3.B.3 (mean value of ankle net force and moment during foot flat )
    % compute the net force at the ankle (F_A) and the net moment at the
    % ankle (M_A) for every timesample during one footflat period
    % <<< ENTER YOUR CODE HERE >>>
    F_A = force(t_TS(1):t_HO(1)) * 1e-3; % conversion into N * m

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
