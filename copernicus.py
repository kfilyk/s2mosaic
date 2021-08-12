# from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import sentinelsat
import glob
import os
from datetime import date
import zipfile

# connect to the API
api = sentinelsat.SentinelAPI('kfilyk', 'wet_garbage')

tiles = ['30NVL', '30NUL', '30NTL', '10UDV', '10UEV', '10UDU', '10UEU', '51RUH', '51RTH', '51RUJ','51RTJ', '48QXH', '48QYJ']
#  ghana : 30VNL, 30NUL, 30NTL
# vancouver/victoria 10UDV, 10UEV, 10UDU, 10UEU
# taiwan: 51RUH, 51RTH, 51RUJ, 51RTJ
# vietnam (hanoi): 48QXH, 48QYJ

query_kwargs = {
    'platformname': 'Sentinel-2',
    'producttype': 'S2MSI2A',  # sentinel 2 multispectrum instrument level 2a
    'date': ('NOW-30DAYS', 'NOW')}  # < - this scans 30 days into the past

# gets the single least cloud covered day from the past 30 days
for tile in tiles:
    kw = query_kwargs.copy()
    kw['filename'] = f'*_T{tile}_*'
    pp = api.query(**kw)

    # collect three days of least cloud cover
    least_clouds_1 = 100  # this is percentage
    least_clouds_2 = 100
    least_clouds_3 = 100
    least_clouds_1_id = ''
    least_clouds_2_id = ''
    least_clouds_3_id = ''

    for p in pp:  # for map in query results
        # if tile has even less cloud cover than the current #1, shuffle #1 to #2
        if pp[p]['cloudcoverpercentage'] < least_clouds_1:
            least_clouds_3_id = least_clouds_2_id
            least_clouds_3 = least_clouds_2
            least_clouds_2_id = least_clouds_1_id
            least_clouds_2 = least_clouds_1
            least_clouds_1_id = p
            least_clouds_1 = pp[p]['cloudcoverpercentage']
            # ['percentwatercoverage'] < 20

    print("#1. %s: %d" % (least_clouds_1_id, least_clouds_1))
    print("#2. %s: %d" % (least_clouds_2_id, least_clouds_2))
    print("#3. %s: %d" % (least_clouds_3_id, least_clouds_3))

    # we have top 3 cloudless days for each tile in 'tiles' list: download to its own folder
    if not os.path.exists('./maps/'+tile+"/"+pp[least_clouds_1_id]['title']+'.SAFE'):
        print(pp[least_clouds_1_id]['title'] +
              " doesn't exist yet. Downloading...")
        api.download(least_clouds_1_id, './maps/'+tile)
    else:
        print(pp[least_clouds_1_id]['title']+" already exists.")

    if not os.path.exists('./maps/'+tile+"/"+pp[least_clouds_2_id]['title']+'.SAFE'):
        print(pp[least_clouds_2_id]['title'] +
              " doesn't exist yet. Downloading...")
        api.download(least_clouds_2_id, './maps/'+tile)
    else:
        print(pp[least_clouds_2_id]['title']+" already exists.")

    if not os.path.exists('./maps/'+tile+"/"+pp[least_clouds_3_id]['title']+'.SAFE'):
        print(pp[least_clouds_3_id]['title'] +
              " doesn't exist yet. Downloading...")
        api.download(least_clouds_3_id, './maps/'+tile)
    else:
        print(pp[least_clouds_3_id]['title']+" already exists.")

    for file in os.listdir():
        print(file)
        if(file.endswith(".zip")):
            with zipfile.ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall('.')
            os.remove(file)

"""
# GeoJSON FeatureCollection containing footprints and metadata of the scenes
api.to_geojson(products)

# GeoPandas GeoDataFrame with the metadata of the scenes and the footprints as geometries
api.to_geodataframe(products)

# Get basic information about the product: its title, file size, MD5 sum, date, footprint and
# its download url
api.get_product_odata(< product_id > )

# Get the product's full metadata available on the server
api.get_product_odata(< product_id > , full=True)
"""
