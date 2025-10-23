from scipy.spatial.transform import Rotation as R
import numpy as np

input_file = r"E:\MA_ADT\room1\agent_0\traj.txt"       # path to your input text file
output_file = r"C:\Users\hthh1\Downloads\room1_agent0.txt"

poses = []
with open(input_file, "r") as f:
    lines = [l.strip() for l in f.readlines() if l.strip()]



num_poses = len(lines)
with open(output_file, "w") as out:
    for i in range(num_poses):
        mat_lines = lines[i]
        line_digit= [float(x) for x in mat_lines.split()]
        mat = np.array([line_digit[i:i+4] for i in range(0, 16, 4)])
        R_mat = mat[:3, :3]
        t_vec = mat[:3, 3]
        q = R.from_matrix(R_mat).as_quat()  # (x, y, z, w)
        out.write(f"{i} {t_vec[0]} {t_vec[1]} {t_vec[2]} {q[0]} {q[1]} {q[2]} {q[3]}\n")

print(f"Converted {num_poses} poses written to {output_file}")


# T=np.array([[ 9.9935108e-001, -1.5576084e-002,3.1508941e-002,-1.2323361e-001	],
#    [  9.2375092e-003,9.8130137e-001,1.9211653e-001,-1.1206967e+000],
#    [ -3.3912845e-002,-1.9170459e-001,9.8083067e-001,9.8870575e-001],
#    [ 0.0000000e+000,0.0000000e+000,0.0000000e+000,1.0000000e+000]])
# R_mat = T[:3, :3]
# r = R.from_matrix(R_mat)
# q = r.as_quat()  # returns (x, y, z, w)
# print(q)