import requests

API_KEY = "a3eac34f9877e8fb484c36f86483636b"

city = "Bangalore"

url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

response = requests.get(url)
data = response.json()

print(data)