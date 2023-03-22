"""
Data loaderers to be used with magrec.
"""

import os
import json

import numpy as np


def dict_2_json(fileName, dictionary, timestamp = True):

    # Save the data
    from json import JSONEncoder
    class NumpyArrayEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return JSONEncoder.default(self, obj)
    
    if timestamp:
        import datetime
        now = datetime.datetime.now()
        fileName = now.strftime("%Y-%m-%d_%H-%M") + fileName

    print("serialize NumPy array into JSON and write into a file")
    with open(fileName + ".json", "w") as write_file:
        json.dump(dictionary, write_file, cls=NumpyArrayEncoder)
    print("Done writing serialized NumPy array into file")
