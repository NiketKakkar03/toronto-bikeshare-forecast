# Data sources, licences, and attribution

Source metadata and licence terms must be verified and recorded when each dataset is first
ingested; catalogue statements can change. Bronze captures retain source URLs, retrieval and feed
times, schema version, content hash, response status, and licence metadata where available.

## Live station feed

Start from the Bike Share Toronto GBFS v3 discovery document:

`https://toronto.publicbikesystem.net/customer/gbfs/v3.0/gbfs.json`

Child endpoints such as `station_information` and `station_status` must be read from discovery,
not guessed or permanently hard-coded. The project plan records the current Mobility Database
catalogue licence as [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); verify it at first
ingestion and indicate changes made to redistributed material.

References:

- [Bike Share Toronto feed catalogue](https://mobilitydatabase.org/feeds/gbfs/gbfs-bike_share_toronto)
- [GBFS specification](https://gbfs.org/get-started/)

## City of Toronto datasets

Historical ridership and other City catalogue data use the licence stated on their dataset page.
The current default is the [Open Government Licence – Toronto](https://open.toronto.ca/open-data-licence/).
Where City data is displayed, include:

> Contains information licensed under the Open Government Licence – Toronto.

## Weather

Historical observations may come from Environment and Climate Change Canada, and forecast or
current observations from MSC GeoMet. Record the terms attached to the exact products used.
Future observed weather is never valid as an input to a historical prediction.

## Independence

This project is not affiliated with or endorsed by the City of Toronto, the Toronto Parking
Authority, or Bike Share Toronto. Do not use public names or logos in a way that implies
endorsement. Forecasts are estimates rather than operational guarantees.

