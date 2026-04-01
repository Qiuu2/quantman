import requests, json
token = 'd75qv5pr01qk56kdpv1gd75qv5pr01qk56kdpv20'

# Check EPS data format from Finnhub
resp = requests.get('https://finnhub.io/api/v1/stock/earnings', params={
    'symbol': 'AAPL', 'token': token
}, timeout=30)
data = resp.json()
print('EPS data type:', type(data))
print('EPS data:', json.dumps(data, indent=2, default=str)[:2000])
