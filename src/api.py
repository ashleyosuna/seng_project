import requests
import re
import time
import csv

BASE_URL = 'https://arctic-shift.photon-reddit.com/'
# TODO: change to actual number
NUM_SAMPLES = 10
START_DATE = '2024-01-01'
VOTE_REGEX = r'(YTA|NTA|ESH|NAH)'

def get_comments(post_id):
    search_params = {'link_id': post_id}

    res = requests.get(BASE_URL + '/api/comments/search', params=search_params)

    if res.status_code != 200:
        print('Error getting comments')
        return []

    comments = res.json()['data']

    return comments

def get_score(comments):
    num_positive = 0
    num_negative = 0

    for comment in comments:
        vote_match = re.search(VOTE_REGEX, comment['body'], re.IGNORECASE)

        if not vote_match:
            continue
        
        ups = comment['ups']

        if vote_match.group() == 'NTA' or vote_match.group() == "NAH":
            num_negative += 1 + ups
        elif vote_match.group() == 'YTA' or vote_match.group() == "ESH":
            num_positive += 1 + ups

    if num_positive < 1 or num_negative < 1:
        return -1

    score = num_positive / (num_negative + num_positive)

    # rounding to two decimal places
    score = (round(score * 100)) / 100
    print(score)
    
    return score


# SEEMS LIKE LIMIT IS 100
# TODO: change to 100
search_params = {'subreddit': 'AmItheAsshole', 'after': START_DATE, 'limit': 10, 'sort': 'asc'}

posts = []
labels = []

while len(posts) < NUM_SAMPLES:
    res = requests.get(BASE_URL + 'api/posts/search', params=search_params)

    if res.status_code != 200:
        print('Error retrieving posts')
        exit(0)

    res_posts = res.json()['data']

    for post in res_posts:
        # ignore posts with no comments or that have been removed
        if post['num_comments'] == None or post['num_comments'] < 10 or post['selftext'] == '[removed]' or post['title'].startswith('AITA Monthly Open Forum'):
            continue
        
        time.sleep(1)
        comments = get_comments(post['id'])
        score = get_score(comments)

        if score != -1:
            title = post['title']
            text = post['selftext']
            
            posts.append(title + ' ' + text)
            labels.append(score)

    # avoid getting repeated posts
    search_params['after'] = res_posts[-1]['created_utc'] + 1

"""
Writes data to csv file.
"""
def write_to_csv(filename, X, y):
    with open(filename, 'w', newline="", encoding="utf-8") as f:
     
        # using csv.writer method from CSV package
        write = csv.writer(f)

        for i in range(len(X)):
            X[i] = X[i].replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
            row = [X[i]] + [y[i]]
            write.writerow(row)

write_to_csv("post_data.csv", posts, labels)

# for post in posts:
#     print(post, '\n\n')