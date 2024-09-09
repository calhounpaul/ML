from PIL import Image
import requests, json, wget, zipfile, random, time, re
from bs4 import BeautifulSoup
import os
import sys
#import pyvips
from urllib.parse import urlparse

#import os, sys, json, re, time, datetime, random, string, docker, atexit

this_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.dirname(this_dir)
permathings_dir = os.path.dirname(scripts_dir)
root_dir = os.path.dirname(permathings_dir)
ephemeral_dir = os.path.join(root_dir, "ephemera")
outputs_dir = os.path.join(ephemeral_dir, "outputs")
cache_dir = os.path.join(ephemeral_dir, "shared_caches/transformers-gpu")
cached_pdfs_and_images_dir = os.path.join(cache_dir, "cached_pdfs_and_images")

libs_dir = os.path.join(permathings_dir, "libs")
sys.path.append(libs_dir)

from selenium_tools import init_selenium_docker, stop_and_remove_selenium_docker, init_selenium_driver_inside_docker

THIS_FILE_PATH = os.path.abspath(__file__)
THIS_FILE_DIR = os.path.dirname(THIS_FILE_PATH)

def grab_page_source(url):
    container = init_selenium_docker()
    time.sleep(3)
    driver = init_selenium_driver_inside_docker()
    driver.get(url)
    time.sleep(5)
    #print page text
    source = driver.page_source
    stop_and_remove_selenium_docker(container)
    return source

def obtain_ikea_pdf_urls():
    source = grab_page_source("https://duckduckgo.com/?q=https%3A%2F%2Fwww.ikea.com%2Fus%2Fen%2Fassembly_instructions%2F&t=h_&ia=web")
    soup = BeautifulSoup(source, "html.parser")
    #find all elements with value data-testid="result-title-a"
    urls = [a["href"] for a in soup.find_all("a", {"data-testid": "result-title-a"}) if a["href"].endswith(".pdf")]
    return urls

def download_ikea_pdfs():
    urls = obtain_ikea_pdf_urls()
    cache_path = cached_pdfs_and_images_dir
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    for url in urls:
        filename = url.split("/")[-1]
        pdf_path = os.path.join(cache_path, filename)
        if not os.path.exists(pdf_path):
            print(f"Downloading {filename}...")
            wget.download(url, pdf_path)
    return cache_path

if __name__ == "__main__":
    download_ikea_pdfs()
    print("Done.")