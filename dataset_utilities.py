import os
import re

metadata_folder_path = 'C:\\Users\\David\\Documents\\228798 Individual Research Project\\dataset\\Originals\\Originals'

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


metadata = load_metadata()
# print(metadata.keys())
answer = metadata_lookup(metadata, '134-03', 'defect', 'braeburn')
print(answer)