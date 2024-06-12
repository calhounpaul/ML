import os, sys, json, re, time, datetime, random, string, docker, atexit
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from io import BytesIO
import tempfile
import html2text

this_modules_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(this_modules_path)
sys.path.append(parent_dir)

from docker_tools import run_docker_container