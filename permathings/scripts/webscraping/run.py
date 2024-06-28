import os, sys, json, re, time, datetime, random, string, docker, atexit

this_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.dirname(this_dir)
permathings_dir = os.path.dirname(scripts_dir)
root_dir = os.path.dirname(permathings_dir)
ephemeral_dir = os.path.join(root_dir, "ephemera")
outputs_dir = os.path.join(ephemeral_dir, "outputs")
libs_dir = os.path.join(permathings_dir, "libs")
sys.path.append(libs_dir)

from selenium_tools import test_tools

test_tools()
