import math
import numpy as np
import cv2
import bisect
import moviepy.editor as mpy
from scipy import ndimage
import subprocess
import random

# 1 Radian equals 57.2958 degrees
RADIANS_TO_DEGREES = 57.2958

# Flickering Constant
FLICKERING_THRESHOLD = 10

# Averaging constant
AVERAGING_CONSTANT = 10

# First and second camera offsets
FIRST_CAMERA_OFFSET = 0
SECOND_CAMERA_OFFSET = 0

# Path to main folder
MAIN_PATH = "C:/Users/mikolez/Desktop/CameraSelection/Instances/"

# Retrieve the video duration and record start timestamp by subtracting duration from the video end timestamp.
# Name of the camera folder is passed as an argument.
def retrieve_video_details(folder_name):
    video_details_file = open(folder_name + "GeneralDataFile.txt", "r")
    video_details_file_contents = video_details_file.read().split(" ")
    video_duration = float(video_details_file_contents[3])
    video_end_utc = float(video_details_file_contents[1])
    video_record_start_utc = video_end_utc - video_duration
    return video_record_start_utc, video_duration

# Retrieve the camera rotation matrix which is responsible for calibration
def retrieve_camera_rotation_matrix(folder_name):
    camera_rotation_matrix = []
    camera_rotation_matrix_file = open(folder_name + "CameraCalibrationFile.txt", "r")
    camera_rotation_matrix_file_contents = camera_rotation_matrix_file.read().split(" ")
    camera_rotation_matrix_file_contents = camera_rotation_matrix_file_contents[:-1]
    for i in range(len(camera_rotation_matrix_file_contents)):
        camera_rotation_matrix.append(float(camera_rotation_matrix_file_contents[i]))
    return camera_rotation_matrix

# Retrieve the data duration and its start timestamp
def retrieve_data_details(folder_name):
    data_details_file = open(folder_name + "dataStartTimeFile.txt", "r")
    data_details_file_contents = data_details_file.read().split(" ")
    data_duration = float(data_details_file_contents[3])
    data_start_utc = float(data_details_file_contents[1])
    return data_start_utc, data_duration

# Retrieve the data rotation matrix
def retrieve_data_rotation_matrix(folder_name):
    data_rotation_matrix = []
    data_rotation_matrix_file = open(folder_name + "rotationMatrixFile.txt", "r")
    data_rotation_matrix_file_contents = data_rotation_matrix_file.read().split("\n")
    data_rotation_matrix_file_contents = data_rotation_matrix_file_contents[:-1]
    for i in range(len(data_rotation_matrix_file_contents)):
        contents_string = data_rotation_matrix_file_contents[i].split(" ")
        contents = []
        for j in contents_string:
            contents.append(float(j))
        data_rotation_matrix.append(contents)
    return data_rotation_matrix

# Retrieve the calibration timestamp relative to the data phone and its rotation matrix through the
# rotation matrix index
def get_calibration_moment_for_data(camera_calibration_moment, data_timestamps):
    ind = bisect.bisect_left(data_timestamps, camera_calibration_moment, 0, len(data_timestamps))
    if ind == 0:
        return data_timestamps[0], 0
    if ind == len(data_timestamps):
        return data_timestamps[len(data_timestamps) - 1], len(data_timestamps) - 1
    diff_left = abs(data_timestamps[ind - 1] - camera_calibration_moment)
    diff_right = abs(data_timestamps[ind] - camera_calibration_moment)
    if (diff_left > diff_right):
        return data_timestamps[ind], ind
    else:
        return data_timestamps[ind - 1], ind - 1

# Retrieve the angle between [0, 0, -1] vectors of camera and data phones
def get_calibration_angle(camera_rotation_matrix, data_rotation_matrix, index, reference_norm):
    camera_vector = [-camera_rotation_matrix[2], -camera_rotation_matrix[5], -camera_rotation_matrix[8]]
    camera_vector = np.array(camera_vector)

    data_vector = [-data_rotation_matrix[index][2], -data_rotation_matrix[index][5], -data_rotation_matrix[index][8]]
    data_vector = np.array(data_vector)

    angle_in_degrees = calc_angle(data_vector, camera_vector)

    normal_vector_1 = cross_product(camera_vector, data_vector)
    normal_vector_1 = np.array(normal_vector_1)

    normal_vector_2 = cross_product(data_vector, camera_vector)
    normal_vector_2 = np.array(normal_vector_2)

    # print(reference_norm)
    # print(normal_vector_1)
    angle_01 = calc_angle(reference_norm, normal_vector_1)
    angle_02 = calc_angle(reference_norm, normal_vector_2)

    if (angle_01 < angle_02):
        normal_vector = normal_vector_1
    else:
        normal_vector = normal_vector_2

    normal_vector = normalize_vector(normal_vector)

    sin_angle = np.dot(cross_product(normal_vector, camera_vector), data_vector)/(np.linalg.norm(data_vector) * np.linalg.norm(camera_vector))

    if (sin_angle < 0):
        angle_in_degrees = 360 - angle_in_degrees

    return angle_in_degrees

# Retrieve the angle between [0, 0, -1] vectors of camera and data phones for moving camera
def get_mov_cam_angle(camera_rotation_matrix, data_rotation_matrix, data_index, camera_index, reference_norm):
    camera_vector = [-camera_rotation_matrix[camera_index][2], -camera_rotation_matrix[camera_index][5], -camera_rotation_matrix[camera_index][8]]
    camera_vector = np.array(camera_vector)

    data_vector = [-data_rotation_matrix[data_index][2], -data_rotation_matrix[data_index][5], -data_rotation_matrix[data_index][8]]
    data_vector = np.array(data_vector)

    angle_in_degrees = calc_angle(data_vector, camera_vector)

    normal_vector_1 = cross_product(camera_vector, data_vector)
    normal_vector_1 = np.array(normal_vector_1)

    normal_vector_2 = cross_product(data_vector, camera_vector)
    normal_vector_2 = np.array(normal_vector_2)

    # print(reference_norm)
    # print(normal_vector_1)
    angle_01 = calc_angle(reference_norm, normal_vector_1)
    angle_02 = calc_angle(reference_norm, normal_vector_2)

    if (angle_01 < angle_02):
        normal_vector = normal_vector_1
    else:
        normal_vector = normal_vector_2

    normal_vector = normalize_vector(normal_vector)

    sin_angle = np.dot(cross_product(normal_vector, camera_vector), data_vector) / (
            np.linalg.norm(data_vector) * np.linalg.norm(camera_vector))

    if (sin_angle < 0):
        angle_in_degrees = 360 - angle_in_degrees

    return angle_in_degrees

# Compute a x b (the cross product)
def cross_product(a, b):
    c = [a[1]*b[2] - a[2]*b[1],
         a[2]*b[0] - a[0]*b[2],
         a[0]*b[1] - a[1]*b[0]]
    return c

# Normalize vector v
def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

# Calculate angle between vectors a and b (not oriented)
def calc_angle(a, b):
    cosine_angle = np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))
    if (cosine_angle < -1):
        cosine_angle = -1
    if (cosine_angle > 1):
        cosine_angle = 1
    angle_in_radians = math.acos(cosine_angle)
    angle_in_degrees = angle_in_radians * RADIANS_TO_DEGREES
    return angle_in_degrees

# Retrieve the frame (by its number) and its timestamp
def get_frame_by_number(cap, number, video_start_utc):
    cap.set(1, number)
    ret, frame = cap.read()
    time = cap.get(cv2.CAP_PROP_POS_MSEC)
    return frame, time + video_start_utc

def get_frame_time(cap, number, video_start_utc):
    cap.set(1, number)
    time = cap.get(cv2.CAP_PROP_POS_MSEC)
    print(time)
    return time + video_start_utc


# Rotate the image by the given angle
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

# Write videos for 2 cameras
def two_cameras_init():
    # Initialization of the main variables
    # Camera 1 details
    camera01_start_utc, camera01_duration = retrieve_video_details(FIRST_CAMERA)
    camera01_rotation_matrix = []
    camera01_rotation_matrix = retrieve_camera_rotation_matrix(FIRST_CAMERA)
    camera01_calibration_moment_utc = camera01_rotation_matrix[-1]
    cap01 = cv2.VideoCapture(FIRST_CAMERA + "video.mp4")

    # Camera 2 details
    camera02_start_utc, camera02_duration = retrieve_video_details(SECOND_CAMERA)
    camera02_rotation_matrix = []
    camera02_rotation_matrix = retrieve_camera_rotation_matrix(SECOND_CAMERA)
    camera02_calibration_moment_utc = camera02_rotation_matrix[-1]
    cap02 = cv2.VideoCapture(SECOND_CAMERA + "video.mp4")

    # Data details
    data_start_utc, data_duration = retrieve_data_details(DATA_PHONE)
    data_rotation_matrix = retrieve_data_rotation_matrix(DATA_PHONE)

    # Calibration angles
    calibration_angle01 = get_calibration_angle(camera01_rotation_matrix, data_rotation_matrix, get_calibration_moment_for_data(camera01_calibration_moment_utc, data_rotation_matrix, data_start_utc)[1])
    calibration_angle02 = get_calibration_angle(camera02_rotation_matrix, data_rotation_matrix, get_calibration_moment_for_data(camera02_calibration_moment_utc, data_rotation_matrix, data_start_utc)[1])

    # Frames and their timestamps of the second video
    camera02_frames = []
    print(int(cap02.get(cv2.CAP_PROP_FRAME_COUNT)))
    for i in range(int(cap02.get(cv2.CAP_PROP_FRAME_COUNT))):
        camera02_frames.append(get_frame_by_number(cap02, i, camera02_start_utc))
        print("Camera 2... ", i)

    # The frame corresponding to the first frame of the second video in terms of time
    synch_timestamp = get_frame_by_number(cap02, 0, camera02_start_utc)[1] - camera01_start_utc
    number = int(synch_timestamp/33.3)

    # Frames and their timestamps of the first video
    camera01_frames = []
    for i in range(501, 3045):
        camera01_frames.append(get_frame_by_number(cap01, i, camera01_start_utc))
        print("Camera 1... ", i)

    # Frames for the final video
    target_frames = []
    for i in range(2544):
        data_index = get_calibration_moment_for_data(camera02_frames[i][1], data_rotation_matrix, data_start_utc)[1]
        camera01_angle = get_calibration_angle(camera01_rotation_matrix, data_rotation_matrix, data_index)
        camera02_angle = get_calibration_angle(camera02_rotation_matrix, data_rotation_matrix, data_index)
        camera01_diff = abs(camera01_angle - calibration_angle01)
        camera02_diff = abs(camera02_angle - calibration_angle02)
        if (camera01_diff <= camera02_diff):
            target_frames.append((camera01_frames[i][0], 1))
        else:
            target_frames.append((camera02_frames[i][0], 2))
        print("Target frames... ", i)

    # Filtering glitches
    for i in range(1, 2543):
        if (target_frames[i][1] != target_frames[i+1][1] and target_frames[i][1] != target_frames[i-1][1]):
            if (target_frames[i][1] == 1):
                target_frames[i] = (camera02_frames[i][0], 2)
            else:
                target_frames[i] = (camera01_frames[i][0], 1)

    # Gray and colored frames initialization
    gray_frames = []
    color_frames = []
    for i in range(2544):
        if (target_frames[i][1] == 1):
            gray_frames.append((camera02_frames[i][0], 2))
            color_frames.append((camera01_frames[i][0], 1))
        else:
            gray_frames.append((camera01_frames[i][0], 1))
            color_frames.append((camera02_frames[i][0], 2))
        print("Gray and Color arrays set up...", i)

    # Make gray frame gray, stitch gray and color images, and add the resulting frame to the array
    final_array = []
    for i in range(2544):
        if (gray_frames[i][1] == 1):
            frame_gray = gray_frames[i][0]
            frame_gray = ndimage.rotate(frame_gray, -90)
            frame_gray = cv2.cvtColor(frame_gray, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

            frame_color = color_frames[i][0]
            frame_color = ndimage.rotate(frame_color, -90)

            vis = np.concatenate((frame_gray, frame_color), axis=1)
            final_array.append(vis)
        else:
            frame_gray = gray_frames[i][0]
            frame_gray = ndimage.rotate(frame_gray, -90)
            frame_gray = cv2.cvtColor(frame_gray, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

            frame_color = color_frames[i][0]
            frame_color = ndimage.rotate(frame_color, -90)

            vis = np.concatenate((frame_color, frame_gray), axis=1)
            final_array.append(vis)
        print("Color and Gray stitching...", i)

    # Extract height and width of the frame for the first video
    cap01.set(1, 0)
    ret, frame = cap01.read()
    frame = ndimage.rotate(frame, -90)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(FOLDER_PATH + "final_video_filtered.mp4", fourcc, 30.0, (width, height))

    for i in range(len(target_frames)):
        frame = ndimage.rotate(target_frames[i][0], -90)
        out.write(frame)
        print("Making the first video... ", i)

    out.release()

    # Extract height and width of the frame for the second video
    height, width, channels = final_array[0].shape

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(FOLDER_PATH + "final_video_filtered_two_views.mp4", fourcc, 30.0, (width, height))

    for i in range(len(final_array)):
        out.write(final_array[i])
        print("Making the second video... ", i)

    out.release()
    print("Done!")

# Write videos for 3 cameras
# n is a folder number in the MAIN_PATH
# is_moving_camera is a flag that indicates if there is a moving camera involved
def three_cameras_init(n, is_moving_camera):

    FOLDER_NUMBER = str(n).zfill(2)
    FOLDER_PATH = MAIN_PATH + FOLDER_NUMBER + "/"

    # Paths to camera and data folders
    FIRST_CAMERA = FOLDER_PATH + "camera01/"
    SECOND_CAMERA = FOLDER_PATH + "camera02/"
    THIRD_CAMERA = FOLDER_PATH + "camera03/"
    DATA_PHONE = FOLDER_PATH + "data/"

    # Initialization of the main variables
    # Camera 1 details
    camera01_start_utc, camera01_duration = retrieve_video_details(FIRST_CAMERA)
    camera01_rotation_matrix = []
    camera01_rotation_matrix = retrieve_camera_rotation_matrix(FIRST_CAMERA)
    camera01_calibration_moment_utc = camera01_rotation_matrix[-1]
    cap01 = cv2.VideoCapture(FIRST_CAMERA + "video.mp4")

    # Camera 2 details
    camera02_start_utc, camera02_duration = retrieve_video_details(SECOND_CAMERA)
    camera02_rotation_matrix = []
    camera02_rotation_matrix = retrieve_camera_rotation_matrix(SECOND_CAMERA)
    camera02_calibration_moment_utc = camera02_rotation_matrix[-1]
    cap02 = cv2.VideoCapture(SECOND_CAMERA + "video.mp4")

    # Camera 3 details
    camera03_start_utc, camera03_duration = retrieve_video_details(THIRD_CAMERA)
    camera03_rotation_matrix = []
    camera03_rotation_matrix = retrieve_camera_rotation_matrix(THIRD_CAMERA)
    camera03_calibration_moment_utc = camera03_rotation_matrix[-1]
    cap03 = cv2.VideoCapture(THIRD_CAMERA + "video.mp4")
    if is_moving_camera:
        camera03_moving_rotation_matrix = retrieve_data_rotation_matrix(THIRD_CAMERA)
        camera03_datastamps = []
        delta = camera03_moving_rotation_matrix[-1][-1] - camera03_duration
        for i in range(len(camera03_moving_rotation_matrix)):
            camera03_datastamps.append(camera03_moving_rotation_matrix[i][-1] - delta + camera03_start_utc)

    # Data details
    data_start_utc, data_duration = retrieve_data_details(DATA_PHONE)
    data_rotation_matrix = retrieve_data_rotation_matrix(DATA_PHONE)
    data_timestamps = []

    for i in range(len(data_rotation_matrix)):
        data_timestamps.append(data_rotation_matrix[i][-1] + data_start_utc)

    # Reference norm initialization
    data_index = get_calibration_moment_for_data(get_frame_by_number(cap03, 0, camera03_start_utc)[1], data_timestamps)[1]
    camera_vector = [-camera03_rotation_matrix[2], -camera03_rotation_matrix[5], -camera03_rotation_matrix[8]]
    camera_vector = np.array(camera_vector)

    data_vector = [-data_rotation_matrix[data_index][2], -data_rotation_matrix[data_index][5], -data_rotation_matrix[data_index][8]]
    data_vector = np.array(data_vector)

    normal_vector = cross_product(camera_vector, data_vector)
    normal_vector = np.array(normal_vector)
    reference_norm = normal_vector

    # Calibration angles
    calibration_angle01 = get_calibration_angle(camera01_rotation_matrix, data_rotation_matrix,
                                                get_calibration_moment_for_data(camera01_calibration_moment_utc,
                                                                                data_timestamps)[
                                                    1], reference_norm)
    calibration_angle02 = get_calibration_angle(camera02_rotation_matrix, data_rotation_matrix,
                                                get_calibration_moment_for_data(camera02_calibration_moment_utc,
                                                                                data_timestamps)[
                                                    1], reference_norm)
    calibration_angle03 = get_calibration_angle(camera03_rotation_matrix, data_rotation_matrix,
                                                get_calibration_moment_for_data(camera03_calibration_moment_utc,
                                                                                data_timestamps)[
                                                    1], reference_norm)

    # The part below sets two constants: FIRST_CAMERA_OFFSET and SECOND_CAMERA_OFFSET.
    # Since the first and the second cameras start earlier, we ignore the their parts in the
    # beginning before the third camera starts for synchronization. FIRST_CAMERA_OFFSET is
    # a frame number which corresponds to the zeroth (0s, or the very first frame) of the
    # third camera by absolute time (chronologically, that is 0s frame of the third video and
    # FIRST_CAMERA_OFFSET frame of the first video were roughly at the same time). The same
    # goes for SECOND_CAMERA_OFFSET
    camera03_first_frame_time = get_frame_by_number(cap03, 0, camera03_start_utc)[1]
    diff_01 = camera03_start_utc - camera01_start_utc
    diff_02 = camera03_start_utc - camera02_start_utc
    ind_01 = int(diff_01 / 33.3)
    ind_02 = int(diff_02 / 33.3)

    min = abs(get_frame_by_number(cap01, ind_01 - 3, camera01_start_utc)[1] - camera03_first_frame_time)
    for i in range(ind_01 - 3, ind_01 + 3):
        time = get_frame_by_number(cap01, i, camera01_start_utc)[1]
        if (abs(time - camera03_first_frame_time) < min):
            min = abs(time - camera03_first_frame_time)
            FIRST_CAMERA_OFFSET = i
            # print(min)
    print("final: ", min)

    min = abs(get_frame_by_number(cap02, ind_02 - 3, camera02_start_utc)[1] - camera03_first_frame_time)
    for i in range(ind_02 - 3, ind_02 + 3):
        time = get_frame_by_number(cap02, i, camera02_start_utc)[1]
        if (abs(time - camera03_first_frame_time) < min):
            min = abs(time - camera03_first_frame_time)
            SECOND_CAMERA_OFFSET = i
            # print(min)
    print("final: ", min)


    # Frames for the final video
    target_frames = []
    camera01_angles = []
    camera02_angles = []
    camera03_angles = []

    # Computing the angles
    for i in range(int(cap03.get(cv2.CAP_PROP_FRAME_COUNT))):
        data_index = get_calibration_moment_for_data(camera03_start_utc + (i + 1) * 33.32, data_timestamps)[1]
        camera01_angle = get_calibration_angle(camera01_rotation_matrix, data_rotation_matrix, data_index, reference_norm)
        camera02_angle = get_calibration_angle(camera02_rotation_matrix, data_rotation_matrix, data_index, reference_norm)
        camera03_angle = get_calibration_angle(camera03_rotation_matrix, data_rotation_matrix, data_index, reference_norm)
        if is_moving_camera:
            camera03_index = get_calibration_moment_for_data(camera03_start_utc + (i + 1) * 33.32, camera03_datastamps)[1]
            camera03_angle = get_mov_cam_angle(camera03_moving_rotation_matrix, data_rotation_matrix, data_index, camera03_index, reference_norm)
        camera01_angles.append(camera01_angle)
        camera02_angles.append(camera02_angle)
        camera03_angles.append(camera03_angle)
    print(camera01_angles)

    # Averaging the angles using AVERAGING_CONSTANT
    i = 0
    while(i < int(cap03.get(cv2.CAP_PROP_FRAME_COUNT))):
        if (i + AVERAGING_CONSTANT >= int(cap03.get(cv2.CAP_PROP_FRAME_COUNT))):
            num = int(cap03.get(cv2.CAP_PROP_FRAME_COUNT)) - i
            avg_angle_01 = sum(camera01_angles[i:]) / num
            avg_angle_02 = sum(camera02_angles[i:]) / num
            avg_angle_03 = sum(camera03_angles[i:]) / num
            for j in range(num):
                camera01_angles[i + j] = avg_angle_01
                camera02_angles[i + j] = avg_angle_02
                camera03_angles[i + j] = avg_angle_03
            i += num
        else:
            avg_angle_01 = sum(camera01_angles[i:i + AVERAGING_CONSTANT]) / AVERAGING_CONSTANT
            avg_angle_02 = sum(camera02_angles[i:i + AVERAGING_CONSTANT]) / AVERAGING_CONSTANT
            avg_angle_03 = sum(camera03_angles[i:i + AVERAGING_CONSTANT]) / AVERAGING_CONSTANT
            for j in range(AVERAGING_CONSTANT):
                camera01_angles[i + j] = avg_angle_01
                camera02_angles[i + j] = avg_angle_02
                camera03_angles[i + j] = avg_angle_03
            i += AVERAGING_CONSTANT
    print(camera01_angles)

    # Comparing angles, choosing the right one, and adding respective frame to target_frames list
    for i in range(int(cap03.get(cv2.CAP_PROP_FRAME_COUNT))):
        camera01_diff = abs(camera01_angles[i] - calibration_angle01)
        camera02_diff = abs(camera02_angles[i] - calibration_angle02)
        camera03_diff = abs(camera03_angles[i] - calibration_angle03)
        # print("Camera 1: ", camera01_diff)
        # print("Camera 2: ", camera02_diff)
        # print("Camera 3: ", camera03_diff)
        # min_diff = min(camera01_diff, camera02_diff, camera03_diff)
        min = camera01_diff
        if (min > camera02_diff):
            min = camera02_diff
        if (min > camera03_diff):
            min = camera03_diff

        if (camera01_diff == min):
            target_frames.append(1)
        elif (camera02_diff == min):
            target_frames.append(2)
        else:
            target_frames.append(3)
        print("Target frames... ", i)

    # Filtering glitches
    # Example: ..., 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
    # both 2s in the above example will be filtered out, since it is clearly flickering (many 1s, then many 3s)
    for i in range(1, len(target_frames) - 1):
        if (target_frames[i] != target_frames[i + 1] and target_frames[i] != target_frames[i - 1]):
            if (target_frames[i-1] == 1):
                target_frames[i] = 1
            elif (target_frames[i-1] == 2):
                target_frames[i] = 2
            else:
                target_frames[i] = 3

    # Filtering using filtering threshold,
    # Example: ..., 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, ...
    # the 2, 2 above will be changed to 1, 1 because it is clearly flickering (there are many 1s on both sides)
    for i in range(1, len(target_frames)):
        if (target_frames[i] != target_frames[i-1]):
            j = i
            count = 0
            while(target_frames[j] == target_frames[i]):
                count += 1
                j += 1
                if (j == len(target_frames)):
                    break
                if (count > FLICKERING_THRESHOLD):
                    break

            if (count <= FLICKERING_THRESHOLD):
                if (i + count < len(target_frames)):
                    if (target_frames[i + count] == target_frames[i - 1]):
                        for k in range(count):
                            target_frames[i + k] = target_frames[i - 1]
                else:
                    for k in range(count):
                        target_frames[i + k] = target_frames[i - 1]

    # When an object turns suddenly from one side camera to the other,
    # the central camera's frames are filtered out below

    # for i in range(1, len(target_frames)):
    #     if (target_frames[i] != target_frames[i-1]):
    #         j = i
    #         count = 0
    #         while(target_frames[j] == target_frames[i]):
    #             count += 1
    #             j += 1
    #             if (j == len(target_frames)):
    #                 break
    #             if (count > 12):
    #                 break
    #
    #         if (count <= 12):
    #             for k in range(int(count/2)):
    #                 target_frames[i + k] = target_frames[i - 1]
    #             for k in range(int(count/2), count):
    #                 target_frames[i + k] = target_frames[i + 12]

    print("Number of frames: ", len(target_frames))



    # Code below is not for camera selection, but rather for creating bad
    # examples of camera selection videos, was used for user study.
    # Feel free to ignore this part.
    # ----------------------------------------------------------------------------------------
    # Get times of turns by our approach
    # turn_indexes = []

    # for i in range(1, len(target_frames)):
    #     if (target_frames[i] != target_frames[i-1]):
    #         turn_indexes.append(i)
    #
    # turn_indexes = np.random.permutation(turn_indexes)
    # turn_indexes = turn_indexes[:int(len(turn_indexes)/2)]
    #
    # for i in range(len(turn_indexes)):
    #     late_or_early = random.randint(1, 3)
    #     if (late_or_early == 1):
    #         for j in range(90):
    #             target_frames[turn_indexes[i] - j] = target_frames[turn_indexes[i] + 1]
    #     else:
    #         for j in range(90):
    #             if (turn_indexes[i] + j >= len(target_frames)):
    #                 break
    #             target_frames[turn_indexes[i] + j] = target_frames[turn_indexes[i] - 1]
    #
    # n_of_flickering = random.randint(1,4)
    # centers = []
    # for i in range(n_of_flickering):
    #     center = random.randint(11, int(cap03.get(cv2.CAP_PROP_FRAME_COUNT)) - 12)
    #     centers.append(center)
    # for i in centers:
    #     j = i - 5
    #     for k in range(10):
    #         rand_view = random.randint(1,4)
    #         target_frames[j + k] = rand_view

    # Extract height and width of the frame for the first video
    # cap01.set(1, 0)
    # ret, frame = cap01.read()
    # frame = np.rot90(frame, -1)
    # height, width, channels = frame.shape
    #
    # cap01.set(1, FIRST_CAMERA_OFFSET)
    # cap02.set(1, SECOND_CAMERA_OFFSET)
    # cap03.set(1, 0)
    #
    # # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    # bad_result = cv2.VideoWriter(FOLDER_PATH + "not_good_result.mp4", fourcc, 30.0, (width, height))
    #
    # for i in range(len(target_frames)):
    #     if (target_frames[i] == 1):
    #         ret, frame01 = cap01.read()
    #         frame01 = np.rot90(frame01, -1)
    #
    #         ret, frame02 = cap02.read()
    #         frame02 = np.rot90(frame02, -1)
    #
    #         ret, frame03 = cap03.read()
    #         frame03 = np.rot90(frame03, -1)
    #
    #         bad_result.write(frame01)
    #     elif (target_frames[i] == 2):
    #         ret, frame01 = cap01.read()
    #         frame01 = np.rot90(frame01, -1)
    #
    #         ret, frame02 = cap02.read()
    #         frame02 = np.rot90(frame02, -1)
    #
    #         ret, frame03 = cap03.read()
    #         frame03 = np.rot90(frame03, -1)
    #
    #         bad_result.write(frame02)
    #     else:
    #         ret, frame01 = cap01.read()
    #         frame01 = np.rot90(frame01, -1)
    #
    #         ret, frame02 = cap02.read()
    #         frame02 = np.rot90(frame02, -1)
    #
    #         ret, frame03 = cap03.read()
    #         frame03 = np.rot90(frame03, -1)
    #
    #         bad_result.write(frame03)
    #     print("Making the video... ", i)
    # bad_result.release()
    # ----------------------------------------------------------------------------------------



    # Extract height and width of the frame for the first video
    cap01.set(1, 0)
    ret, frame = cap01.read()
    frame = np.rot90(frame, -1)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case

    out = cv2.VideoWriter(FOLDER_PATH + "final_video_filtered.mp4", fourcc, 30.0, (width, height))
    out02 = cv2.VideoWriter(FOLDER_PATH + "final_video_filtered_two_views.mp4", fourcc, 30.0, (3 * width, height))

    # Creating videos of three cameras
    # Feel free to ignore, was used for paper
    # camera01_out = cv2.VideoWriter(FOLDER_PATH + "camera01.mp4", fourcc, 30.0, (width, height))
    # camera02_out = cv2.VideoWriter(FOLDER_PATH + "camera02.mp4", fourcc, 30.0, (width, height))
    # camera03_out = cv2.VideoWriter(FOLDER_PATH + "camera03.mp4", fourcc, 30.0, (width, height))

    # Setting the first and second videos at the correct frame, since they
    # started earlier than the third video in order to make sync videos
    cap01.set(1, FIRST_CAMERA_OFFSET)
    cap02.set(1, SECOND_CAMERA_OFFSET)
    cap03.set(1, 0)

    # Choosing the frame to write by looking at target_frames
    for i in range(len(target_frames)):
        if (target_frames[i] == 1):
            ret, frame01 = cap01.read()
            frame01 = np.rot90(frame01, -1)

            ret, frame02 = cap02.read()
            frame02 = np.rot90(frame02, -1)

            ret, frame03 = cap03.read()
            frame03 = np.rot90(frame03, -1)

            # Turning frames to black and white (gray colored)
            frame02 = cv2.cvtColor(frame02, cv2.COLOR_BGR2GRAY)
            frame02 = cv2.cvtColor(frame02, cv2.COLOR_GRAY2BGR)

            frame03 = cv2.cvtColor(frame03, cv2.COLOR_BGR2GRAY)
            frame03 = cv2.cvtColor(frame03, cv2.COLOR_GRAY2BGR)

            # Frames concatenation
            vis = np.concatenate((frame01, frame02), axis=1)
            vis = np.concatenate((vis, frame03), axis=1)

            # camera01_out.write(frame01)
            # camera02_out.write(frame02)
            # camera03_out.write(frame03)

            out.write(frame01)
            out02.write(vis)
            print("Case: ", 1)

        elif (target_frames[i] == 2):
            ret, frame01 = cap01.read()
            frame01 = np.rot90(frame01, -1)

            ret, frame02 = cap02.read()
            frame02 = np.rot90(frame02, -1)

            ret, frame03 = cap03.read()
            frame03 = np.rot90(frame03, -1)

            # Turning frames to black and white (gray colored)
            frame01 = cv2.cvtColor(frame01, cv2.COLOR_BGR2GRAY)
            frame01 = cv2.cvtColor(frame01, cv2.COLOR_GRAY2BGR)

            frame03 = cv2.cvtColor(frame03, cv2.COLOR_BGR2GRAY)
            frame03 = cv2.cvtColor(frame03, cv2.COLOR_GRAY2BGR)

            # Frames concatenation
            vis = np.concatenate((frame01, frame02), axis=1)
            vis = np.concatenate((vis, frame03), axis=1)

            # camera01_out.write(frame01)
            # camera02_out.write(frame02)
            # camera03_out.write(frame03)

            out.write(frame02)
            out02.write(vis)
            print("Case: ", 2)

        else:
            ret, frame01 = cap01.read()
            frame01 = np.rot90(frame01, -1)

            ret, frame02 = cap02.read()
            frame02 = np.rot90(frame02, -1)

            ret, frame03 = cap03.read()
            frame03 = np.rot90(frame03, -1)

            # Turning frames to black and white (gray colored)
            frame02 = cv2.cvtColor(frame02, cv2.COLOR_BGR2GRAY)
            frame02 = cv2.cvtColor(frame02, cv2.COLOR_GRAY2BGR)

            frame01 = cv2.cvtColor(frame01, cv2.COLOR_BGR2GRAY)
            frame01 = cv2.cvtColor(frame01, cv2.COLOR_GRAY2BGR)

            # Frames concatenation
            vis = np.concatenate((frame01, frame02), axis=1)
            vis = np.concatenate((vis, frame03), axis=1)

            # camera01_out.write(frame01)
            # camera02_out.write(frame02)
            # camera03_out.write(frame03)

            out.write(frame03)
            out02.write(vis)
            print("Case: ", 3)

        print("Making the videos... ", i)

    out.release()
    out02.release()
    # camera01_out.release()
    # camera02_out.release()
    # camera03_out.release()
    print("Done!")

three_cameras_init(16, True)

# Below is a ffmpeg command to retrieve sound from a video
# command = "ffmpeg -i C:/Users/mikolez/Desktop/CameraSelection/Instances/07/camera03/video.mp4 -ab 160k -ac 2 -ar 44100 -vn C:/Users/mikolez/Desktop/CameraSelection/Instances/07/audio.wav"
# subprocess.call(command, shell=True)

# Below is a ffmpeg command to write sound to a video
# cmd = 'ffmpeg -y -i C:/Users/mikolez/Desktop/CameraSelection/Instances/16_good/cut_sound.wav  -r 30 -i C:/Users/mikolez/Desktop/CameraSelection/Instances/16_good/final_video_filtered_two_views.mp4  -filter:a aresample=async=1 -c:a flac -c:v copy av.mkv'
# subprocess.call(cmd, shell=True)                                     # "Muxing Done
