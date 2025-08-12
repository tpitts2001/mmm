import nasdaqdatalink

api_key = "B1Wh18SoAX5kHoKi1tWE"
data = nasdaqdatalink.Database('AAPL').bulk_download_url()
#nasdaqdatalink.Database('AAPL').bulk_download_to_file('test_data/AAPL.csv')

print(data)