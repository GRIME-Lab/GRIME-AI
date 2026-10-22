#!/usr/bin/env python3
# Probe the ICOS Carbon Portal so we can finalize the ICOS sensor (variable) query.
# Prints: ecosystem stations, one station's level-2 data objects, and that object's
# variable columns (name + unit). Needs only requests + pandas.
# Run:  python icos_probe.py     then paste the output.

import requests
import pandas as pd

SPARQL = "https://meta.icos-cp.eu/sparql"


def sparql(query):
    r = requests.post(SPARQL, data=query.encode("utf-8"),
                      headers={"Accept": "application/json", "Content-Type": "text/plain"}, timeout=60)
    r.raise_for_status()
    d = r.json()
    cols = d["head"]["vars"]
    return pd.DataFrame([{c: b.get(c, {}).get("value") for c in cols}
                         for b in d["results"]["bindings"]], columns=cols)


# 1) Ecosystem (ES) stations
STATIONS = """PREFIX cpmeta: <http://meta.icos-cp.eu/ontologies/cpmeta/>
SELECT DISTINCT ?uri ?id ?name ?stationTheme ?lat ?lon
FROM <http://meta.icos-cp.eu/resources/icos/>
FROM <http://meta.icos-cp.eu/resources/cpmeta/>
WHERE { ?uri cpmeta:hasStationId ?id . ?uri a ?stationTheme .
  OPTIONAL { ?uri cpmeta:hasName ?name }
  OPTIONAL { ?uri cpmeta:hasLatitude ?lat }
  OPTIONAL { ?uri cpmeta:hasLongitude ?lon } }
ORDER BY ?id"""
st = sparql(STATIONS)
st["theme"] = st["stationTheme"].astype(str).str.rsplit("/", n=1).str[-1]
es = st[st["theme"] == "ES"].reset_index(drop=True)
print("ICOS ecosystem (ES) stations:", len(es))
print(es[["id", "name", "lat", "lon"]].head(12).to_string(index=False))

if es.empty:
    raise SystemExit("No ES stations returned; paste the above and stop here.")

# 2) Level-2 data objects for the first ES station
row = es.iloc[0]
DOBJ = """prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>
prefix cpmeta: <http://meta.icos-cp.eu/ontologies/cpmeta/>
prefix prov: <http://www.w3.org/ns/prov#>
select ?dobj ?specLabel ?timeStart ?timeEnd where {
  ?spec cpmeta:hasDataLevel ?dl . FILTER(?dl = 2)
  ?spec rdfs:label ?specLabel .
  VALUES ?station { <%s> }
  ?dobj cpmeta:hasObjectSpec ?spec .
  FILTER NOT EXISTS {[] cpmeta:isNextVersionOf ?dobj}
  ?dobj cpmeta:wasAcquiredBy [ prov:startedAtTime ?timeStart ; prov:endedAtTime ?timeEnd ;
                               prov:wasAssociatedWith ?station ] .
} ORDER BY DESC(?timeEnd)""" % row["uri"]
do = sparql(DOBJ)
print(f"\nStation {row['id']} ({row['name']}) level-2 data objects: {len(do)}")
print(do[["specLabel", "timeStart", "timeEnd"]].head(12).to_string(index=False))

# 3) Variable columns of the first data object (this is the sensor list)
if not do.empty:
    pid = do.iloc[0]["dobj"]
    meta = requests.get(pid + "/meta.json", headers={"Accept": "application/json"}, timeout=60).json()
    cols = meta.get("specificInfo", {}).get("columns", [])
    print(f"\nVariables in '{do.iloc[0]['specLabel']}'  ({len(cols)}):")
    for v in cols[:50]:
        unit = (v.get("valueType", {}) or {}).get("unit", "")
        print("  ", v.get("label"), (f"[{unit}]" if unit else ""))
