import os
from itertools import chain, islice
from distutils import util
import hashlib

def prepare_folders(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)

def to_chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))

def tobool(input):
    try:
        return bool(util.strtobool(input))
    except:
        return False

def toNone(input):
    if str(input).lower() in ['', 'none', 'null']:
        return None
    else:
        return input

def parse_size(size=None, def_size='640,480'):
    if size is None:
        size = def_size
    size_lst = list(map(int, size.split(',')))
    return size_lst


def colorize_log(string, color):
    colors = dict(
        grey="\x1b[38;21m",
        yellow="\x1b[33;21m",
        red="\x1b[31;21m",
        bold_red="\x1b[31;1m",
        green="\x1b[32;1m",
    )
    reset = "\x1b[0m"
    col = colors.get(color)
    if col is None:
        return string
    string = f"{col}{string}{reset}"
    return string

def validate_max_size(max_size):
    if max_size[0] % 32 != 0 or max_size[1] % 32 != 0:
        max_size[0] = max_size[0] // 32 * 32
        max_size[1] = max_size[1] // 32 * 32
    return max_size

def check_hash(filename, hash, algo='md5'):
    """Check whether hash of the file content matches the expected hash.
    Parameters
    ----------
    filename : str
        Path to the file.
    hash : str
        Expected hash in hexadecimal digits.
    algo: str
        Hashing algorithm (md5, sha1, sha256, sha512)

    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    algos = {
        'md5': hashlib.md5,
        'sha1': hashlib.sha1,
        'sha256': hashlib.sha256,
        'sha512': hashlib.sha512,
    }
    hasher = algos[algo]()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            hasher.update(data)

    file_hash = hasher.hexdigest()
    l = min(len(file_hash), len(hash))
    return hasher.hexdigest()[0:l] == hash[0:l]