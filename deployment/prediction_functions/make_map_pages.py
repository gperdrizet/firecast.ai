import pandas as pd
from datetime import datetime


def make_maps(predictions: 'DataFrame', today: str, heatmaps_dir: str) -> bool:
    daily_predictions = predictions.groupby(['day'])
    dates = predictions.day.unique()
    print(dates)

    for day, data in daily_predictions:
        print(f'{day}:\n {data.head()}')

        if day == today:
            output_file = f'{heatmaps_dir}index.html'
        else:
            output_file = f'{heatmaps_dir}{day}.html'

        output_header = f"""
        <!DOCTYPE html>
        <html>

        <head>
            <title>Firecast.ai</title>
            <meta name="viewport" content="width=device-width">
            <script src="https://polyfill.io/v3/polyfill.min.js?features=default"></script>
            <script
                src="https://maps.googleapis.com/maps/api/js?key=AIzaSyDIMl2IOesPgOHI8zxMhLErq4VZQvw0gx8&callback=initMap&libraries=visualization&v=weekly"
                defer></script>
            <style type="text/css">
                /* Always set the map height explicitly to define the size of the div
            * element that contains the map. */
                #map {{
                    height: 100%;
                }}

                /* Optional: Makes the sample page fill the window. */
                html,
                body {{
                    height: 100%;
                    margin: 0;
                    padding: 0;
                }}

                #top-floating-panel {{
                    font: 400 11px Roboto, Arial, sans-serif;
                    margin: 10px;
                    z-index: 5;
                    position: absolute;
                    cursor: pointer;
                    right: 0%;
                    top: 10%;
                    width: 110px;
                    height: 160px
                }}

                .button {{
                    cursor: pointer;
                    -webkit-user-drag: none;
                    font-weight: 400;
                    padding-top: 10px;
                    direction: ltr;
                    overflow: hidden;
                    text-align: center;
                    height: 30px;
                    width: 110px;
                    vertical-align: middle;
                    color: rgb(86, 86, 86);
                    font-family: Roboto, Arial, sans-serif;
                    user-select: none;
                    font-size: 18px;
                    background-color: rgb(255, 255, 255);
                    border-bottom-right-radius: 2px;
                    border-bottom-left-radius: 2px;
                    background-clip: padding-box;
                    box-shadow: rgba(0, 0, 0, 0.5) 0px 1px 4px -1px;
                    border-bottom: 0px;
                }}

                a:link, a:visited, a:hover, a:active {{
                    text-decoration: none;
                    font: 400 11px Roboto, Arial, sans-serif;
                    color: rgb(86, 86, 86);
                    font-size: 18px;
                }}

            </style>
            <script>
                "use strict";

                // This example requires the Visualization library. Include the libraries=visualization
                // parameter when you first load the API. For example:
                // <script src="https://maps.googleapis.com/maps/api/js?key=YOUR_API_KEY&libraries=visualization">

        let map, heatmap;

        function initMap() {{
            map = new google.maps.Map(document.getElementById("map"), {{
                center: {{
                    lat: 36.7,
                    lng: -119
                }},
                minZoom: 1,
                maxZoom: 9.25,
                zoom: 1,
                restriction: {{
                    latLngBounds: {{
                        east: -100,
                        north: 45.3,
                        south: 25,
                        west: -140
                    }},
                    //32.436919, -114.036252
                    strictBounds: true
                }},
                mapTypeId: "roadmap"
            }});

            heatmap = new google.maps.visualization.HeatmapLayer({{
                data: getPoints(),
                map: map,
                dissipating: false,
                radius: 0.5
            }});
        }}

        function toggleHeatmap() {{
            heatmap.setMap(heatmap.getMap() ? null : map);
        }}

                function changeGradient() {{
                    const gradient = [
                        "rgba(0, 255, 255, 0)",
                        "rgba(0, 255, 255, 1)",
                        "rgba(0, 191, 255, 1)",
                        "rgba(0, 127, 255, 1)",
                        "rgba(0, 63, 255, 1)",
                        "rgba(0, 0, 255, 1)",
                        "rgba(0, 0, 223, 1)",
                        "rgba(0, 0, 191, 1)",
                        "rgba(0, 0, 159, 1)",
                        "rgba(0, 0, 127, 1)",
                        "rgba(63, 0, 91, 1)",
                        "rgba(127, 0, 63, 1)",
                        "rgba(191, 0, 31, 1)",
                        "rgba(255, 0, 0, 1)"
                    ];
                    heatmap.set("gradient", heatmap.get("gradient") ? null : gradient);
                }}

                function changeOpacity() {{
                    heatmap.set("opacity", heatmap.get("opacity") ? null : 0.2);
                }}

                function getPoints() {{

                    return ["""

        script_footer = f"""
                ];
                        }}

                    </script>
                </head>

                <body>
                    <div id="top-floating-panel">"""

        google_map = f"""
                    </div> 
                    <div id="map"></div>
                </body>
                </html>"""

        with open(output_file, 'w') as file:
            file.write(output_header)

            for index, row in data.iterrows():
                file.write(
                    f'\n{{ location: new google.maps.LatLng({row["lat"]}, {row["lon"]}), weight: {row["ignition_risk"]} }},')

            file.write(script_footer)

            for date in dates:
                day_num = pd.to_datetime(date).day
                month = pd.to_datetime(date).month

                if date == day:
                    if date == today:
                        file.write(
                            f'<div class="button"><a href="index.html"><b>{pd.to_datetime(date).day_name()[0:3]} {month}/{day_num}</b></a></div>')

                    else:
                        file.write(
                            f'<div class="button"><a href="{date}.html"><b>{pd.to_datetime(date).day_name()[0:3]} {month}/{day_num}</b></a></div>')
                else:
                    if date == today:
                        file.write(
                            f'<div class="button"><a href="index.html">{pd.to_datetime(date).day_name()[0:3]} {month}/{day_num}</a></div>')

                    else:
                        file.write(
                            f'<div class="button"><a href="{date}.html">{pd.to_datetime(date).day_name()[0:3]} {month}/{day_num}</a></div>')

            file.write(google_map)
