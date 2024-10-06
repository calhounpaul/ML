import wget, os, sys, time
import re, requests, os

#an estonian comedy podcast
rss_url = "https://www.omnycontent.com/d/playlist/ad35d8cb-2dfe-45c8-a4f9-a68700d12423/f4d84dd7-5be7-41e6-9cd2-a68700d1647e/365fdde8-dbf8-4f2a-be23-a68700d300c1/podcast.rss"

ephemera_folder = "/ephemera"
output_folder = os.path.join(ephemera_folder, 'podcast_files')
os.makedirs(output_folder, exist_ok=True)

if not os.path.exists(os.path.join(ephemera_folder, "podcast.xml")):
    rss_content = requests.get(rss_url).content
    with open(os.path.join(ephemera_folder, "podcast.xml"), "wb") as f:
        f.write(rss_content)
else:
    with open(os.path.join(ephemera_folder, "podcast.xml"), "rb") as f:
        rss_content = f.read()

from bs4 import BeautifulSoup

rss_filepath = os.path.join(ephemera_folder, 'podcast.xml')

with open(rss_filepath, 'r') as f:
    rss_text = f.read()

soup = BeautifulSoup(rss_text, 'xml')

episodes = soup.find_all('item')

for episode in episodes:
    title = episode.find('title').text
    permalink = episode.find('link').text
    filepath_mp3 = episode.find('enclosure')['url']
    filename = permalink.split('/')[-1] + '.mp3'
    if not os.path.exists(os.path.join(output_folder, filename)):
        print('Downloading ' + filename)
        file_content = requests.get(filepath_mp3).content
        tmp_filepath = os.path.join(output_folder, filename + '.tmp')
        with open(tmp_filepath, 'wb') as f:
            f.write(file_content)
        os.rename(tmp_filepath, os.path.join(output_folder, filename))
    else:
        print('Skipping ' + filename)
print('Done')