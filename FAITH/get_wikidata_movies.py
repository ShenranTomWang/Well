from dotenv import load_dotenv

load_dotenv()

import argparse
import json
import os

import requests


def run_sparql(sparql_query):
    wikidata_endpoint = "https://query.wikidata.org/sparql"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/json",
    }

    response = requests.get(wikidata_endpoint, params={"query": sparql_query, "format": "json"}, headers=headers)
    if response.status_code == 200:
        try:
            data = response.json()
            return data["results"]["bindings"]
        except Exception as e:
            print(e)
            return None

    print("Error:", response.status_code)
    return None


def query_movie_publication(limit_num, offset=0):
    sparql_query = f"""
        SELECT ?movie ?movieLabel ?time ?language ?languageLabel ?director ?directorLabel
        WHERE
        {{
          ?movie wdt:P31 wd:Q11424.
          ?movie wdt:P577 ?time.
          ?movie wdt:P364 ?language.
          ?movie wdt:P57 ?director.
          FILTER(?language IN (wd:Q1860))
          FILTER(STRSTARTS(STR(?movie), "http://www.wikidata.org/entity/Q"))
          FILTER(STRSTARTS(STR(?language), "http://www.wikidata.org/entity/Q"))
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".}}
        }}
        LIMIT {limit_num}
        OFFSET {offset}
    """
    return run_sparql(sparql_query)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query English-language Wikidata movies and write one JSON object per line."
    )
    parser.add_argument(
        "--result_file",
        required=True,
        help="Path to the output JSONL file.",
    )
    parser.add_argument(
        "--limit_num",
        type=int,
        default=10000,
        help="Maximum number of Wikidata bindings to request. Defaults to 10000.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="SPARQL OFFSET to start from. Defaults to 0.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite result_file instead of appending to it.",
    )
    return parser.parse_args()


def write_movie_results(result_file, triples, overwrite=False):
    from dateutil import parser
    from tqdm import tqdm

    result_dir = os.path.dirname(os.path.abspath(result_file))
    os.makedirs(result_dir, exist_ok=True)
    mode = "w" if overwrite else "a"

    outputs = []
    prev_movie = None
    prev_result = None

    with open(result_file, mode) as f:
        pbar = tqdm(triples)
        for triple in pbar:
            movie = triple["movieLabel"]["value"]
            release_year = parser.isoparse(triple["time"]["value"]).year
            if movie == prev_movie:
                if release_year not in prev_result["time"]:
                    prev_result["time"].append(release_year)
                continue

            if prev_result is not None:
                pbar.set_description(f"Saved {len(outputs) + 1} samples")
                outputs.append(prev_result)
                f.write(json.dumps(prev_result) + "\n")

            prev_movie = movie
            prev_result = {
                "movie": movie,
                "time": [release_year],
                "director": triple["directorLabel"]["value"],
                "info": triple,
            }

        if prev_result is not None:
            outputs.append(prev_result)
            f.write(json.dumps(prev_result) + "\n")

    return outputs


def main():
    args = parse_args()
    triples = query_movie_publication(args.limit_num, offset=args.offset) or []
    triples = [triple for triple in triples if triple is not None]
    outputs = write_movie_results(args.result_file, triples, overwrite=args.overwrite)
    print(f"Finished Running! Wrote {len(outputs)} samples to {args.result_file}")


if __name__ == "__main__":
    main()
