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

def init_selenium_docker(driver_port=4444, vnc_port=7900):
    container = run_docker_container("selenium/standalone-firefox", {f"{driver_port}/tcp": driver_port, f"{vnc_port}/tcp": vnc_port}, {}, {}, detach=True)
    return container

def stop_and_remove_selenium_docker(container):
    container.stop()
    container.remove()

def init_selenium_driver_inside_docker(driver_port=4444):
    options = Options()
    driver = webdriver.Remote(command_executor=f"http://localhost:{driver_port}/wd/hub", options=options)
    return driver

def test_tools():
    container = init_selenium_docker()
    time.sleep(3)
    driver = init_selenium_driver_inside_docker()
    driver.get("https://yahoo.com")
    time.sleep(5)
    #print page text
    source = driver.page_source
    #soup = BeautifulSoup(source, "html.parser")
    #text = soup.get_text()
    text = html2text.html2text(source)
    print(text)
    stop_and_remove_selenium_docker(container)
    print("Test done")
