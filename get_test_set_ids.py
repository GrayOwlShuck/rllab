import os, sys
import glob
import subprocess
import re

test_files = glob.glob('vendor/mujoco_models/test_objs/*.stl')

test_names = [fname[fname.index('test_objs')+10:] for fname in test_files]

ids = {}

for name in test_names:
    output = subprocess.check_output('grep ' + name + ' /home/cfinn/code/rllab/vendor/local_mujoco_models/ensure_w*distractor_pusher*', shell=True)
    files = re.findall('[0-9]*.xml', output)
    name_ids = [f[:-4] for f in files]
    for id in name_ids:
        ids[id] = True

keys = ids.keys()
keys.sort()
keys = [str(int(key) - 1) for key in keys]
print(keys)
print(len(keys))


