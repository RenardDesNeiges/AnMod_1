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