import json

def read_json(path):
    """
    Read a .json file

    Args:
        path (string): Path of a .json file

    Returns:
         data (dictionary): Output dictionary
    """
    
    f = open(path,)
    data = json.load(f)
    f.close()
    
    return data

def write_json(files, path, indent=4):
    """
    Write a json file from a dictionary

    Args:
        files (dictionary): Data
        path (string): Saved json path
        indent (int, optional): Number of spaces of indentation. Defaults to 4.
    """
    
    json_object = json.dumps(files, indent = indent) 

    # Writing to saved_path_json
    with open(path, "w") as outfile: 
        outfile.write(json_object) 