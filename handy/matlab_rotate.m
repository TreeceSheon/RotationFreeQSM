function img_rot = rotate(img, z_prjs)

    rot_mat = RotMat([0;0;1], z_prjs');
    imsize = size(img);
    [X, Y, Z] = ndgrid(-imsize(1)/2:imsize(1)/2-1,-imsize(2)/2:imsize(2)/2-1,-imsize(3)/2:imsize(3)/2-1);
    new_coords = rot_mat * [X(:),Y(:),Z(:)]';
    new_coords = reshape(new_coords,[3, imsize(1), imsize(2), imsize(3)]);
    x = squeeze(new_coords(1,:,:,:));
    y = squeeze(new_coords(2,:,:,:));
    z = squeeze(new_coords(3,:,:,:));
    img_rot = interpn(X,Y,Z, img, x,y,z);
    
end


function [U] = RotMat(A, B)
    % rotation from unit vector A to B; 
    % return rotation matrix U such that UA = B;
    % and ||U||_2 = 1
        if A == B
            U = [[1, 0, 0]
                 [0, 1, 0]
                 [0, 0, 1]];
    
        else
            GG = @(A,B) [ dot(A,B) -norm(cross(A,B)) 0;
                          norm(cross(A,B)) dot(A,B)  0;
                          0              0           1];
    
            FFi = @(A,B) [ A (B-dot(A,B)*A)/norm(B-dot(A,B)*A) cross(B,A) ];
    
            UU = @(Fi,G) Fi*G*inv(Fi);
    
    
            U = UU(FFi(A,B), GG(A,B));
        end
    end
    
    
    

