import requests
import re
import time
import utils

BASE_URL = 'https://arctic-shift.photon-reddit.com/'
# TODO: change to actual number
NUM_SAMPLES = 1000
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
search_params = {'subreddit': 'AmItheAsshole', 'after': START_DATE, 'limit': 100, 'sort': 'asc'}

posts = []
labels = []
last_date = None

try:
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

        print('date of last post retrieved', res_posts[-1]['created_utc'])
        last_date = res_posts[-1]['created_utc']

        # avoid getting repeated posts
        search_params['after'] = last_date + 1
        time.sleep(1)
    csv_rows = []
    for i in range(len(posts)):
        csv_rows.append([posts[i], labels[i]])

    utils.write_to_csv(csv_rows)
    print('num of posts retrieved', len(posts))

except:
    print('an error occurred: last date', last_date)
    csv_rows = []
    for i in range(len(posts)):
        csv_rows.append([posts[i], labels[i]])
    utils.write_to_csv(csv_rows)
    print('num of posts retrieved', len(posts))

# for post in posts:
#     print(post, '\n\n')