import ee

# Initialize the Earth Engine API
ee.Initialize()

print(ee.String('Hello from the Earth Engine servers!').getInfo())
