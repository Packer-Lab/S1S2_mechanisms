import os
import json
import sys
import getpass, socket
from pathlib import Path

def loadpaths(username=None, hostname=None):
    # Where is this file?
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    json_path = os.path.join(Path(__location__).parent.parent, 'data_paths.json')
    json_path = str(json_path)

    if username is None:
        username = getpass.getuser()  # get username of PC account
    if hostname is None: 
        hostname = socket.gethostname() 
    assert type(username) == str and type(hostname) == str 
    name_use = f'{username}_{hostname}'

    with open(json_path, 'r') as config_file:
        config_info = json.load(config_file)
        user_paths_dict = config_info[name_use]['paths']  # extract paths from current user

    # Expand tildes in the json paths
    # user_paths_dict = {k: str(v) for k, v in user_paths_dict.items()}
    return {k: os.path.expanduser(v) for k, v in user_paths_dict.items()}
