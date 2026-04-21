# This file intentionally contains code quality issues for demonstration purposes.
# The Code Analyzer agent will detect these problems.

import os
import sys
import json
from math import *   # wildcard import — bad practice

SECRET_KEY = "mysecretpassword123"   # hardcoded secret — security issue
API_KEY = "sk-abc123def456"          # hardcoded API key


def process_user_data(data, items=[], config={}):
    # mutable default arguments — anti-pattern
    result = eval(data)   # dangerous eval — security vulnerability
    return result


def calculate(a, b, c, d, e, f, g):
    # too many parameters (7)
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:   # deeply nested — 5 levels
                        return a + b + c + d + e
    return 0


def fetch_data(url):
    # no docstring
    import urllib.request
    data = urllib.request.urlopen("http://example.com/api")   # HTTP not HTTPS
    return data.read()


def parse_input(raw_input):
    # bare except
    try:
        return json.loads(raw_input)
    except:
        return None


def very_long_function_that_does_too_many_things(x):
    # this function is intentionally long
    result = x * 2
    result = result + 1
    result = result - 0
    result = result * 1
    result = result / 1
    result = result + 0
    result = result - 1
    result = result + 3
    result = result * 2
    result = result - 2
    result = result + 5
    result = result - 3
    result = result + 7
    result = result - 4
    result = result + 9
    result = result - 5
    result = result + 11
    result = result - 6
    result = result + 13
    result = result - 7
    result = result + 15
    result = result - 8
    result = result + 17
    result = result - 9
    result = result + 2
    result = result + 2
    result = result + 2
    result = result + 2
    result = result + 2
    result = result + 2
    result = result + 2
    return result


class DataProcessor:
    # no class docstring
    def __init__(self, name, config, handler, logger, formatter, validator):
        # too many parameters in __init__
        self.name = name

    def run(self):
        pass


import pickle
import hashlib

def insecure_hash(data):
    return hashlib.md5(data.encode()).hexdigest()   # weak hash — MD5


def load_data(file_path):
    with open(file_path, "rb") as f:
        return pickle.loads(f.read())   # unsafe pickle deserialization
