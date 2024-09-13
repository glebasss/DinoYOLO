import cv2
import os
import uuid


def delete_files_in_directory(directory_path):
    try:
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("All files deleted successfully.")
    except OSError:
        print("Error occurred while deleting files.")


video_folder = r'my-video/video.mp4'
save_folder = r'screens-from-video'
delete_files_in_directory(save_folder)

cap = cv2.VideoCapture(video_folder)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % fps == 0:
        save_path = os.path.join(save_folder, str(uuid.uuid4()) + '.jpg')
        print(frame_count/30, 'saved successfully')
        cv2.imwrite(save_path, frame)
    frame_count += 1
print('Finish')
cap.release()
