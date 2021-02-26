function Out=applyLowpassFilter(InputSignal,CutoffFrequency, SamplingFrequency)
% This function applies a lowpass butterworth filter to the input signal
% provide the InputSignal, CutoffFrequency and SamplingFrequency as input
    [b,a] = butter(2, 1.2465*CutoffFrequency/SamplingFrequency*2, 'low');
    Out= filtfilt(b,a,InputSignal);
end

