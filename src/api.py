import requests
import re
import time

BASE_URL = 'https://arctic-shift.photon-reddit.com/'
# TODO: change to actual number
NUM_SAMPLES = 10
START_DATE = '2024-01-01'
VOTE_REGEX = r'(YTA|NTA)'

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

        if vote_match.group() == 'NTA':
            num_negative += 1 + ups
        elif vote_match.group() == 'YTA':
            num_positive += 1 + ups

    if num_positive == 0 and num_negative == 0:
        return -1

    score = num_positive / (num_negative + num_positive)

    # rounding to two decimal places
    score = (round(score * 100)) / 100
    
    return score


# SEEMS LIKE LIMIT IS 100
# TODO: change to 100
search_params = {'subreddit': 'AmItheAsshole', 'after': START_DATE, 'limit': 1, 'sort': 'asc'}

posts = []

while len(posts) < NUM_SAMPLES:
    res = requests.get(BASE_URL + 'api/posts/search', params=search_params)

    if res.status_code != 200:
        print('Error retrieving posts')
        exit(0)

    res_posts = res.json()['data']

    for post in res_posts:
        # ignore posts with no comments or that have been removed
        if post['num_comments'] == None or post['num_comments'] == 0 or post['selftext'] == '[removed]':
            continue
        
        time.sleep(1)
        comments = get_comments(post['id'])
        score = get_score(comments)

        if score != -1:
            title = post['title']
            text = post['selftext']

            data = {'title': title, 'text': text, 'label': score}
            
            posts.append(data)

    # avoid getting repeated posts
    search_params['after'] = res_posts[-1]['created_utc'] + 1
    time.sleep(1)

# for post in posts:
#     print(post, '\n\n')


