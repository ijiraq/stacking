import mp_ephem
import sys


def app():
    print(mp_ephem.BKOrbit(None, sys.argv[1]).summarize())
