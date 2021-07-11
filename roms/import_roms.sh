CURDIR=$(pwd)
cd $(pip list -v | grep atari-py | awk '{print $3}')/atari_py/
python -m atari_py.import_roms $CURDIR
cd $CURDIR