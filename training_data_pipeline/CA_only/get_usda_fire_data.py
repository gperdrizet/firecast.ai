import requests
from zipfile import ZipFile


def get_fire_data(url: str, path: str, file_name: str) -> 'DataFrame':
    zip_archive = f'{path}{file_name}'
    stream = requests.get(url, stream=True)

    with open(zip_archive, 'wb') as output:
        for chunk in stream.iter_content(chunk_size=128):
            output.write(chunk)

    with ZipFile(zip_archive, 'r') as zip_file:
        zip_file.printdir()

        filename = 'Data/FPA_FOD_20170508.sqlite'

        try:
            data = zip_file.read(filename)
        except KeyError:
            print(f'ERROR: Did not find {filename} in zip file')
        else:
            print
            print(repr(data))
