import numpy as np

Image_points=np.array([[757,213],[758,415],[758,686],[759,966],[1190,172],[329,1041],[1204,850],[340,159]])
World_points=np.array([[0,0,0],[0,3,0],[0,7,0],[0,11,0],[7,1,0],[0,11,7],[7,9,0],[0,1,7]])

def length_vec(x):
    return  np.linalg.norm(x)
#M=K.Rot
M=[]
for i in range(len(World_points)):
    x_w, y_w,z_w= World_points[i]
    u, v = Image_points[i]
    M.append([x_w, y_w,z_w, 1, 0, 0, 0,0, -u*x_w, -u*y_w,-u*z_w, -u])
    M.append([0, 0, 0,0, x_w, y_w,z_w, 1, -v*x_w, -v*y_w,-v*z_w, -v])
M = np.array(M)
_, _, V = np.linalg.svd(M)
Projection_mat = V[-1].reshape(3, 4)
Projection_mat_normalized = Projection_mat / Projection_mat[-1, -1]
Q, R = np.linalg.qr(Projection_mat_normalized.T)
Projection_mat_normalized = Q.T

# Alternatively, you might ensure that the last row of the matrix is [0, 0, 0, 1]
Projection_mat_normalized[-1] = [0, 0, 0, 1]
print("Projection Matrix for Q1:\n",Projection_mat)
print("Normalized Projection Matrix is", Projection_mat_normalized)

# print(Projection_mat.shape)
"""
We will slice the Projection matrix to get A_dash matrix 
that will be used to split the (3x3) matrix intoQ and R matrix using Gram Schmidt method 
"""

Sliced_Matrix=Projection_mat[0:3,0:3]
print()
#H=Rot.K
#H(inv)=Rot(inv).K(inv)
#Decompose H(inv) in Q and R basically, K(inv) is our R and Q inv is our Rot (inv)

# reference-https://www.youtube.com/watch?v=oFZQykvEw14 
H= np.linalg.inv(Sliced_Matrix)
a=H[0:3,0]
b=H[0:3,1]
c=H[0:3,2]
# print(c)
#Now we will calculate the orthonormal vectors
#QR decomposition using gram schmidt
#reference -https://www.youtube.com/watch?v=TRktLuAktBQ&list=LL&index=2 and wikipedia
q1=a/length_vec(a)
q2_prime=b-((np.dot(b,q1))*q1)
q2=q2_prime/length_vec(q2_prime)
q3_prime=c-((((np.dot(c,q1))*q1)))-((((np.dot(c,q2))*q2)))
q3=q3_prime/length_vec(q3_prime)
#stacking q1,q2,q3
Q=np.array([q1,q2,q3]).T
#We have got the Q matrix which is [q1,q2,q3], which is column of orthonormal vectors
# For getting the the R(upper triangular matrix we will calculate)
q1_norm=q1/length_vec(q1)
q2_norm=q2/length_vec(q2)
q3_norm=q3/length_vec(q3)
Q_norm=np.array([q1_norm,q2_norm,q3_norm]).T

#Rotation matrix is Q_norm (inv)
Rotation=np.linalg.inv(Q_norm)
print("Rotation Matrix for Q1:\n",Rotation)

"""
we know Hinv=Rot(inv)K(inv)multiply Q.T on both sides
Q.T Hinv= Q Q.T K(inv)
Q.T Hinv=K(inv)
"""
R=Q_norm.T@np.linalg.inv(Sliced_Matrix)
K=np.linalg.inv(R)
print("Intrinsic Matrix for Q1:\n",K)
print(Projection_mat[0:3,3])
Translation= np.dot((np.linalg.inv(K)),Projection_mat[:,-1])
print("Translation Matrix for Q1:\n",Translation)
reprojection_errors = []
for i in range(len(World_points)):
    X, Y, Z = World_points[i]
    u, v = Image_points[i]
    world_pt_homogeneous = np.array([X, Y, Z, 1])
    projected_pt_homogeneous = Projection_mat @ world_pt_homogeneous
    projected_pt = projected_pt_homogeneous[:2] / projected_pt_homogeneous[2]
    reprojection_error = np.sqrt((u - projected_pt[0]) ** 2 + (v - projected_pt[1]) ** 2)
    reprojection_errors.append(reprojection_error)
    

# Print the reprojection errors
print("Reprojection errors:")
print(reprojection_errors)
print("Mean Error:",np.mean(reprojection_errors))

