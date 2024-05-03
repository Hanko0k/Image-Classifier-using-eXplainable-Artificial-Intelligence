import os
import re
import shutil
from sklearn.model_selection import train_test_split

metadata_folder_path = 'C:\\Users\\David\\Documents\\228798 Individual Research Project\\dataset\\Originals\\Originals'
og_dataset = "C:\\Users\\David\\Documents\\228798 Individual Research Project\\dataset\\1 Braeburn\\1 Braeburn"

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

    base_dest_dir = "braeburn binary classifier dataset"

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


# metadata = load_metadata()
# generate_file_structure(['defective', 'no defect'])
# print(metadata.keys())
# answer = metadata_lookup(metadata, '002-22', 'defect', 'braeburn')
# print(answer)
# verify_directory_same_type("C:\\Users\\David\\Documents\\Image-Classifier-using-eXplainable-Artificial-Intelligence\\braeburn binary classifier dataset\\defective")