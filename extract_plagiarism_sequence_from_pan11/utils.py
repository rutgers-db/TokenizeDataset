import xml.etree.ElementTree as ET
import os

def extract_plagiarism_features(xml_file):
    """
    Extract plagiarism features from an XML file.

    Args:
    xml_file (str): Path to the XML file.

    Returns:
    list of dict: A list of dictionaries, each containing plagiarism feature attributes.
    """
    features = []
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for feature in root.findall('feature'):
        if feature.get('name') == 'plagiarism':
            features.append(feature.attrib)

    return features


def extract_sequence_from_file(file_path, offset, length):
    """
    Extract a specific sequence from a text file.

    Args:
    file_path (str): Path to the text file.
    offset (int): The offset where the sequence starts.
    length (int): The length of the sequence.

    Returns:
    str: The extracted sequence.
    """
    # with open(file_path, 'r', encoding='utf-8') as file:
    #     file.seek(offset)
    #     sequence = file.read(length)
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
        sequence = file_content[offset:offset + length]
    return sequence

def extract_txt_filenames(root_dir):
    """
    Extract all .txt file names under all subdirectories of a given root directory.

    Args:
    root_dir (str): The root directory to search in.

    Returns:
    dict: A dictionary mapping file names to their full paths.
    """
    txt_files = {}
    for subdir in next(os.walk(root_dir))[1]:  # Iterate through each subdirectory
        subdir_path = os.path.join(root_dir, subdir)
        for file in os.listdir(subdir_path):  # List files in each subdirectory
            if file.endswith('.txt'):
                full_path = os.path.join(subdir_path, file)
                txt_files[file] = full_path
    return txt_files


def replace_str_suffix(str, new_suffix='.txt'):
    """
    Change the file's suffix to a new suffix.

    Args:
    str (str): Path to the file.
    new_suffix (str): New file suffix (default is '.txt').

    Returns:
    str: New file path with the updated suffix.
    """
    if not os.path.isfile(str):
        print(f"Error: The file '{str}' does not exist.")
        return None

    # Split the file path into directory, name, and extension
    dir_name, file_name = os.path.split(str)
    name, ext = os.path.splitext(file_name)

    if ext.lower() != '.xml':
        print(f"Error: The file '{str}' is not an XML file.")
        return None

    # Create the new file name with the new suffix
    new_file_name = name + new_suffix
    new_file_path = os.path.join(dir_name, new_file_name)

    return new_file_path

def jaccard_similarity(arr1, arr2):
    # Convert arrays to sets to get unique elements
    set1 = set(arr1)
    set2 = set(arr2)

    # Calculate the intersection and union
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    # Calculate Jaccard similarity
    # We use float division (with `len(union)` as a denominator) to handle division by zero
    similarity = len(intersection) / float(len(union))

    return similarity

def get_file_content(filepath):
     with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
        return content
