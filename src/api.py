import requests
import re

BASE_URL = 'https://arctic-shift.photon-reddit.com/'

search_params = {'subreddit': 'AmItheAsshole', 'after': '2024-01-01', 'before': '2024-01-30', 'limit': '10'}

res = requests.get(BASE_URL + 'api/posts/search', params=search_params)

posts = res.json()['data']

# FILTERING OF POSTS


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

        for comment in comments:
            text = comment['body']
        
            m = re.search('(NTA|YTA)', text)

            if m:
                label = labels[m.group()]
                post_data = {'title': post['title'], 'text': post['selftext'], 'label': label}
                data.append(post_data)
                break

print(data)