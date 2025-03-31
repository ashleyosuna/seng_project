import requests
import re
import time
import csv
from datetime import datetime, timedelta
import urllib.parse


BASE_URL = 'https://arctic-shift.photon-reddit.com/'
VOTE_REGEX = r'(YTA|NTA|ESH|NAH)'
START_DATE = datetime(2022, 6, 25)
END_DATE = datetime(2024, 12, 31)
CSV_FILENAME = f"{START_DATE.strftime('%Y-%m-%d')}_to_{END_DATE.strftime('%Y-%m-%d')}.csv"

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
    return score

# Function to write posts to a CSV file after each iteration
def write_to_csv(data, filename):
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)

# Loop through dates, each day
current_date = START_DATE
while current_date <= END_DATE:
    # Convert the date to ISO 8601 format
    start_date_iso = current_date.strftime('%Y-%m-%dT00:00')
    end_date_iso = (current_date - timedelta(minutes=1) + timedelta(days=1)).strftime('%Y-%m-%dT%H:%M')

    search_params = {
        'subreddit': 'AmItheAsshole',
        'after': start_date_iso,
        'before': end_date_iso,
        'limit': 100,
        'sort': 'asc'
    }

    posts = []
    labels = []

    try:
        # Get posts for the current day
        res = requests.get(BASE_URL + 'api/posts/search', params=search_params)

        if res.status_code != 200:
            print('Error retrieving posts')
            continue

        res_posts = res.json()['data']

        for post in res_posts:
            # Ignore posts with no comments, removed posts, or less than 10 comments
            if post['num_comments'] is None or post['num_comments'] < 10 or post['selftext'] == '[removed]' or post['title'].startswith('AITA Monthly Open Forum'):
                continue

            comments = get_comments(post['id'])
            score = get_score(comments)

            if score != -1:
                title = post['title']
                text = post['selftext']

                # Append post and score to lists
                posts.append([title + ' ' + text, score])

        # Write the posts and labels to CSV after each day iteration
        write_to_csv(posts, CSV_FILENAME)
        print(f"Posts from {start_date_iso} to {end_date_iso} written to CSV.")

        # Move to the next day
        current_date += timedelta(days=1)

    except Exception as e:
        print(f'An error occurred: {e}')
        continue
