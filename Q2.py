import os
import numpy as np
import cv2

# List of image paths
List_of_images = ["Calibrate/image1.jpg", "Calibrate/image2.jpg", "Calibrate/image3.jpg", "Calibrate/image4.jpg", "Calibrate/image5.jpg", "Calibrate/image6.jpg", "Calibrate/image7.jpg", "Calibrate/image8.jpg", "Calibrate/image9.jpg", "Calibrate/image10.jpg", "Calibrate/image11.jpg", "Calibrate/image12.jpg", "Calibrate/image13.jpg"]

# Criteria for corner detection refinement
end_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.0003)

# Create the world coordinates of the chessboard corners
World_Points = np.zeros((9*6, 3), np.float32)
World_Points[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
World_Points *= 21.5

# Lists to store world and image plane coordinates
World_Coordinates = [] 
Image_plane_Coordinates = []

# Create the "Results" folder if it doesn't exist
os.makedirs("Results", exist_ok=True)

# Loop through each image
for idx, file in enumerate(List_of_images):
    # Read the image
    image = cv2.imread(file)
    grayed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(grayed_image, (9,6), None)
    
    # If corners are found
    if ret: 
        # Append world coordinates
        World_Coordinates.append(World_Points)
        
        # Refine corner positions
        Corners_prime = cv2.cornerSubPix(grayed_image, corners, (9,9), (-1,-1), end_criteria)
        Image_plane_Coordinates.append(Corners_prime)
        
        # Draw and save the image with corners
        cv2.drawChessboardCorners(image, (9,6), Corners_prime, ret)
        cv2.imwrite(f"Results/image{idx+1}_corners.jpg", image)
        
        # Display the image with corners for visualization (optional)
        cv2.namedWindow("Corners_prime", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Corners_prime", 300, 300)
        cv2.imshow("Corners_prime", image)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Perform camera calibration
ret, k_matrix, dis, rotation, translation = cv2.calibrateCamera(World_Coordinates, Image_plane_Coordinates, grayed_image.shape[::-1], None, None)
print("Intrinsic Matrix (K):\n", k_matrix)

# Calculate reprojection errors
Total_err = 0 
for i in range(len(World_Coordinates)):
    Reprojection, _ = cv2.projectPoints(World_Coordinates[i], rotation[i], translation[i], k_matrix, dis)
    error = cv2.norm(Image_plane_Coordinates[i], Reprojection, cv2.NORM_L2) / len(Reprojection)
    print("Error in reprojection for image", i+1, ".jpg is:", error) 
    Total_err += error 

# Calculate mean reprojection error
mean_error = Total_err / len(World_Coordinates) 
print("Mean error in reprojection:", mean_error)
