from internetarchive import download
import sys
import os
from py7zr import unpack_7zarchive
import shutil

arg_plus_glob = sys.argv[1] + ".stackexchange.com.7z"
arg_extract = sys.argv[1] + "_extract"

download('stackexchange', verbose=True, glob_pattern=arg_plus_glob)

for item in os.listdir("./stackexchange"):

    if item.startswith(sys.argv[1]):
        print(item)

        try:
            os.mkdir("./stackexchange/" + item )
        except:
            pass

        shutil.register_unpack_format('7zip', ['.7z'], unpack_7zarchive)
        shutil.unpack_archive('./stackexchange/'+item, "./data/stackexchange_raw/" + item)

