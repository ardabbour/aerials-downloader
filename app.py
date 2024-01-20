import json
import requests
import tqdm
import urllib3
import os
import os.path
import sqlite3
import subprocess
from concurrent.futures import ThreadPoolExecutor
urllib3.disable_warnings()

json_file_path='/Library/Application Support/com.apple.idleassetsd/Customer/entries.json'
aerial_folder_path='/Library/Application Support/com.apple.idleassetsd/Customer/4KSDR240FPS/'

def getAerials(path):
    aerialsList = [{}]
    with open(path) as f:
        d = json.load(f)
        for aerial in d["assets"]:
            aerialsList.append(aerial)
    return aerialsList


def downloadAerial(url: str, file_path: str, name:str):
    with open(file_path, 'wb') as f:
        with requests.get(url, stream=True,verify=False) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))

            # tqdm has many interesting parameters. Feel free to experiment!
            tqdm_params = {
                'desc': name,
                'total': total,
                'miniters': 1,
                'unit': 'B',
                'unit_scale': True,
                'unit_divisor': 1024,
            }
            with tqdm.tqdm(**tqdm_params) as pb:
                for chunk in r.iter_content(chunk_size=8192):
                    pb.update(len(chunk))
                    f.write(chunk)


def updateSQL():
    con = sqlite3.connect("/Library/Application Support/com.apple.idleassetsd/Aerial.sqlite")
    cur = con.cursor()
    cur.execute("VACUUM;")
    cur.execute("UPDATE ZASSET SET ZLASTDOWNLOADED = 718364962.0204;")
    con.commit()
    con.close()

def killService():
    #idleassetsd
    subprocess.run(["killall", "idleassetsd"]) 

def downloadAerialsParallel(aerial):
    if 'url-4K-SDR-240FPS' in aerial:
        url = aerial["url-4K-SDR-240FPS"].replace('\\', '')
        file_path = aerial_folder_path + aerial["id"] + '.mov'
        if not os.path.exists(file_path):
            print("Start download of " + aerial["accessibilityLabel"])
            downloadAerial(url, file_path, aerial["accessibilityLabel"])

def downloadAerials():
    aerials = getAerials(json_file_path)
    
    # Get the number of download threads from the environment variable
    download_threads = int(os.environ.get('DOWNLOAD_THREADS', 1))

    with ThreadPoolExecutor(max_workers=download_threads) as executor:
        executor.map(downloadAerialsParallel, aerials)

print("Loading Aerials list")
downloadAerials()
print("Updating Aerials Database")
updateSQL()
print("Restarting service")
killService()