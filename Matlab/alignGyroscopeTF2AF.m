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