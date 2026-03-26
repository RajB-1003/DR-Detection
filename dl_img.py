import requests
url = "https://upload.wikimedia.org/wikipedia/commons/e/e0/Fundus_photograph_of_normal_left_eye.jpg"
response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
with open('test_normal.jpg', 'wb') as f:
    f.write(response.content)
print("Normal eye image downloaded!")
