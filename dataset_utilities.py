import os
import re
import shutil
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from PIL import Image

metadata_folder_path = 'C:\\Users\\David\\Documents\\228798 Individual Research Project\\dataset\\Originals\\Originals'
og_dataset = "C:\\Users\\David\\Documents\\228798 Individual Research Project\\dataset\\3 Granny Smith"

def _clean_filename(name: str) -> str:
    name = os.path.splitext(name)[0]
    name = name.strip()
    name = re.sub(r'^[\d\W_]+|[\d\W_]+$', '', name)
    return name

def load_metadata() -> dict:
    file_names = [
        file for file in os.listdir(metadata_folder_path)
        if os.path.isfile(os.path.join(metadata_folder_path, file)) and file != 'Key to tables.txt'
    ]

    metadata = {_clean_filename(name).lower(): [] for name in file_names}

    for file_name, name in zip(file_names, metadata):
        file_path = os.path.join(metadata_folder_path, file_name)
        with open (file_path, 'r') as f:
            for line in f:
                metadata[name].append(line.strip())

    return metadata

def metadata_lookup(metadata: dict, img_name: str, attribute: str, apple_species: str=None) -> str:
    original_pattern = r'^\d{3}-\d{2}$'
    augmented_pattern = r'^\d{2}-\d{3}-\d{2}$'
    name, _ = os.path.splitext(img_name) # If file extension, remove

    if re.match(original_pattern, name) is not None:
        avaliable_attributes = ['defect']

        if attribute not in avaliable_attributes:
            raise ValueError(f"Query Error: '{attribute}' attribute not recognised")
        
        elif apple_species.lower() not in metadata:
            raise ValueError(f"Query Error: '{apple_species}' apple species not recognised")
        
        else:
            result = None
            name = name.replace('-', '')
            for value in metadata[apple_species]:
                if name == value[:5]:
                    result = value
                    break
            if result == None:
                raise ValueError(f"Query Error: No metadata found for {img_name} in {apple_species}")
            
            if attribute == 'defect':
                if result[5] == '3':
                    return 'defective'
                else:
                    return 'no defect'
            

    elif re.match(augmented_pattern, img_name) is not None:
        attributes = ['species', 'defect']

        if attribute not in attributes:
            raise ValueError(f"Queried attribute '{attribute}' not recognised")
        

# def get_image_attribute(metadata: dict, img_name: str, attribute: str, apple_species: str=None) -> str:
#     original_pattern = r'^\d{3}-\d{2}$'
#     augmented_pattern = r'^\d{2}-\d{3}-\d{2}$'

#     return value

def _get_attribute(attr: str) -> str:

    return None

def augment_metadata() -> None:

    return None

def generate_file_structure(classes: list) -> None:

    base_dest_dir = "granny smith binary classifier dataset"

    train_dir = os.path.join(base_dest_dir, 'train')
    val_dir = os.path.join(base_dest_dir, 'validation')
    test_dir = os.path.join(base_dest_dir, 'test')

    for dir in [train_dir, val_dir, test_dir]:
        os.makedirs(dir, exist_ok=True)
        for c in classes:
            os.makedirs(os.path.join(dir, c), exist_ok=True)

    img_list = [f for f in os.listdir(og_dataset) if os.path.isfile(os.path.join(og_dataset, f))]

    train, temp = train_test_split(img_list, test_size=0.4, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    
    _move_files(train, train_dir)
    _move_files(val, val_dir)
    _move_files(test, test_dir)

    return None

def _move_files(file_names, dest_dir: str):
    extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    for name in file_names:

        _, ext = os.path.splitext(name)
        if ext.lower() not in extensions:
            continue

        label = metadata_lookup(metadata, name, 'defect', 'braeburn')
        shutil.move(os.path.join(og_dataset, name), os.path.join(dest_dir, label, name))

def verify_directory_same_type(dir: str)-> None:
    file_names = os.listdir(dir)
    types_found = set()

    for name in file_names:
        types_found.add(metadata_lookup(metadata, name, 'defect', 'braeburn'))

    print(f"INFO: {len(types_found)} types found in directory:")
    for type in types_found:
        print(type)

    return None

def prepend_string_to_images(directory, prepend_str):
    """
    Prepend a given string to all image files in the specified directory.

    Args:
    directory (str): Path to the directory containing the image files.
    prepend_str (str): String to prepend to each image file name.

    Returns:
    None: Files are renamed in place.
    """
    # Define possible image file extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is an image
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            # Construct the new filename
            new_filename = prepend_str + filename
            # Construct full old and new file paths
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            # Rename the file
            os.rename(old_file, new_file)

def remove_white(image):
   # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the range for white color in HSV
    lower_white = np.array([0, 0, 0], dtype=np.uint8)
    upper_white = np.array([180, 70, 255], dtype=np.uint8)

    lower_white = np.array([0, 0, 0], dtype=np.uint8)
    upper_white = np.array([180, 70, 255], dtype=np.uint8)
    
    # Create a mask for the white color
    mask = cv2.inRange(hsv_image, lower_white, upper_white)
    
    # Invert the mask to keep only the non-white parts
    inverted_mask = cv2.bitwise_not(mask)
    
    # Apply the mask to the original image
    result_image = cv2.bitwise_and(image, image, mask=inverted_mask)
    
    return result_image

def remove_background(image):
    # Step 1: Load the BMP image
    # image = cv2.imread(image_path)
    
    # Step 2: Color Thresholding
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # # Define range for apple colors in HSV
    # lower_red1 = np.array([0, 50, 50])
    # upper_red1 = np.array([10, 255, 255])
    # lower_red2 = np.array([170, 50, 50])
    # upper_red2 = np.array([180, 255, 255])
    # lower_green = np.array([40, 40, 40])
    # upper_green = np.array([70, 255, 255])
    # lower_yellow = np.array([20, 100, 100])
    # upper_yellow = np.array([30, 255, 255])

        # # Define range for apple colors in HSV
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    lower_green = np.array([30, 100, 100])
    upper_green = np.array([80, 255, 255])

    lower_yellow = np.array([10, 40, 80])
    upper_yellow = np.array([30, 255, 255])
    
    # Threshold the HSV image to get only apple colors
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.bitwise_or(mask_red1, mask_red2)
    mask = cv2.bitwise_or(mask, mask_green)
    mask = cv2.bitwise_or(mask, mask_yellow)
    
    # Step 3: Find contours and create mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    apple_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Adjust area threshold as needed
            cv2.drawContours(apple_mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    # Debugging Step: Save the initial apple mask
    # cv2.imwrite('apple_initial_mask.bmp', apple_mask)
    
    # Step 4: Close gaps inside the apple mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # closed_mask = cv2.morphologyEx(apple_mask, cv2.MORPH_CLOSE, kernel)
    closed_mask = cv2.erode(apple_mask, kernel, iterations=2)
    # closed_mask = cv2.GaussianBlur(closed_mask, (9, 9), 0)
    closed_mask = cv2.medianBlur(closed_mask, 31)
    # closed_mask = cv2.dilate(apple_mask, kernel, iterations=1)
    
    # Debugging Step: Save the closed mask
    # cv2.imwrite('apple_closed_mask.bmp', apple_mask)
    
    # Step 5: Apply the mask to the original image
    result = cv2.bitwise_and(image, image, mask=closed_mask)
    
    return result

# image_with_contours = process_image("C:\\Users\\David\\Documents\\Image-Classifier-using-eXplainable-Artificial-Intelligence\\Datasets\\103-17.bmp")
# cv2.imwrite("C:\\Users\\David\\Documents\\Image-Classifier-using-eXplainable-Artificial-Intelligence\\Datasets\\test_2.bmp", image_with_contours)

def preprocess_images_from_directory(target_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for f in os.listdir(target_dir):
        f_path = os.path.join(target_dir, f)

        if os.path.isfile(f_path) and f_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # with Image.open(f_path) as img:
            img = cv2.imread(f_path)
    
            processed_img = remove_background(img)
            # processed_img = remove_white(img)

            output_path = os.path.join(dest_dir, f)
            # processed_img.save(output_path)
            cv2.imwrite(output_path, processed_img)

    return None

TARGET = "C:\\Users\\David\\Documents\\Image-Classifier-using-eXplainable-Artificial-Intelligence\\Datasets\\Mixed Binary\\no defect"
DEST = "C:\\Users\\David\\Documents\\Image-Classifier-using-eXplainable-Artificial-Intelligence\\Datasets\\Augmented Mixed Binary\\no defect"
preprocess_images_from_directory(TARGET, DEST)




# metadata = load_metadata()
# generate_file_structure(['defective', 'no defect'])
# prepend_string_to_images("C:\\Users\\David\\Documents\\Image-Classifier-using-eXplainable-Artificial-Intelligence\\Datasets\\Mixed Binary\\Good Granny", '03-')
# print(metadata.keys())
# answer = metadata_lookup(metadata, '002-22', 'defect', 'braeburn')
# print(answer)
# verify_directory_same_type("C:\\Users\\David\\Documents\\Image-Classifier-using-eXplainable-Artificial-Intelligence\\braeburn binary classifier dataset\\defective")