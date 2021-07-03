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