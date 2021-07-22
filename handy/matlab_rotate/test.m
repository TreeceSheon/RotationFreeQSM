delta_angle = pi / 8;
cube = padarray(ones([100, 100, 100]), [78, 78, 78], 'both');

for angle_z = 1:4
    for angle_xy = 1:4
        z_prj = [cos(delta_angle * angle_xy) * sin(delta_angle * angle_z), ...
                           sin(delta_angle * angle_xy) * sin(delta_angle * angle_z), cos(delta_angle * angle_z)];
        cube_rot = rotate(cube, z_prj);
        niftiwrite(cube_rot, ['mat_cube', num2str(angle_z), num2str(angle_xy), '.nii']);
    end
end
