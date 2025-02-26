import requests
import re
import time
import datetime

BASE_URL = 'https://arctic-shift.photon-reddit.com/'
# TODO: change to actual number
MIN_NUM_POSTS = 200
START_DATE = int((datetime.datetime(2024, 1, 1) - datetime.datetime(1970, 1, 1)) / datetime.timedelta(seconds=1))

# SEEMS LIKE LIMIT IS 100
search_params = {'subreddit': 'AmItheAsshole', 'after': START_DATE, 'limit': 100, 'sort': 'asc'}

posts = []

while len(posts) < MIN_NUM_POSTS:
    res = requests.get(BASE_URL + 'api/posts/search', params=search_params)

    if res.status_code != 200:
        print('Error retrieving posts')
        exit(0)
    
    posts += res.json()['data']

    # ensure we do not get repeated posts
    last_date = posts[-1]['created_utc']
    search_params['after'] = last_date + 1

    # rate limiting for the api
    time.sleep(1)

# filter out posts that have their content removed
posts = [post for post in posts if post['selftext'] != '[removed]']

# FILTERING OF POSTS

# KEEP X% MOST POPULAR POSTS
PERCENTAGE = 0.1
posts = sorted(posts, key=lambda d: d['ups'], reverse=True)[:round(len(posts) * 0.1)]

# CLEAN UP TEXT? REMOVE NEW LINES AND OTHER CHARACTERS

# LABELING DATA
data = []
labels = {'NTA': 0, 'YTA': 1}
for post in posts:
    id = post['id']

    search_params = {'link_id': id}

    res = requests.get(BASE_URL + '/api/comments/search', params=search_params)

    if (res.status_code == 200):
        comments = res.json()['data']

        comments = sorted(comments, key=lambda d: d['ups'], reverse=True)

        # ignore posts with no comments or comments without any upvotes
        if len(comments) == 0 or comments[0]['ups'] == None or comments[0]['ups'] == 0:
            break

        for comment in comments:
            text = comment['body']
        
            m = re.search('(NTA|YTA)', text)

            if m:
                label = labels[m.group()]
                post_data = {'title': post['title'], 'text': post['selftext'], 'label': label}
                data.append(post_data)
                break

    time.sleep(1)

