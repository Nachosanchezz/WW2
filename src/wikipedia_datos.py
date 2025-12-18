import requests
import json
import time
from pathlib import Path

# Si ya tienes config.py, mejor importa DATA_PROCESSED de ahí:
from config import DATA_PROCESSED

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
WIKI_FILE = DATA_PROCESSED / "wiki_docs.jsonl"

# IMPORTANTE: pon aquí algo identificable tuyo (recomendado por Wikipedia)
HEADERS = {
    "User-Agent": "Ignacio-WW2/0.1 (contact: isanccal@myuax.com)"
}

KEYWORDS = [
    "World War II",
    "Invasion of Poland",
    "Battle of France",
    "Battle of Britain",
    "Operation Sea Lion",
    "Operation Barbarossa",
    "Operation Typhoon",
    "Operation Torch",
    "Operation Husky",
    "Operation Overlord",
    "Operation Bagration",
    "Operation Market Garden",
    "Pearl Harbor",
    "Battle of Moscow",
    "Battle of Stalingrad",
    "Battle of Kursk",
    "D-Day",
    "Normandy landings",
    "Battle of Midway",
    "Battle of Guadalcanal",
    "Battle of Iwo Jima",
    "Battle of Okinawa",
    "Winston Churchill",
    "Franklin D. Roosevelt",
    "Joseph Stalin",
    "Adolf Hitler",
    "Benito Mussolini",
    "Hideki Tojo",
    "Nazi Germany",
    "Fascist Italy",
    "Imperial Japan",
    "Allies of World War II",
    "Axis powers",
    "Holocaust",
    "Nazi concentration camps",
    "Final Solution",
    "War crimes in World War II",
    "Nuremberg trials",
    "German war economy",
    "British war economy",
    "American war production",
    "Manhattan Project",
    "Atomic bombings of Hiroshima and Nagasaki",
    "Firebombing of Tokyo",
    "European theatre of World War II",
    "Pacific War",
    "Eastern Front (World War II)",
    "Western Front (World War II)",
    "North African Campaign (World War II)",
    "Italian Campaign (World War II)",
    "Post–World War II",
    "Consequences of World War II",
    "Division of Germany",
    "Cold War origins",
    "United Nations",

    # --- EXTRA KEYWORDS CLAVE ---
    "Heinrich Himmler",
    "Reinhard Heydrich",
    "Joseph Goebbels",
    "Hermann Göring",
    "SS",
    "Gestapo",
    "Red Army",
    "Royal Air Force",
    "Molotov–Ribbentrop Pact",
    "Yalta Conference",
    "Potsdam Conference",
    "Tehran Conference",
    "Luftwaffe",
    "Panzer divisions",
    "Battle of El Alamein",
    "Battle of the Bulge",
    "Attack on Pearl Harbor",
    "Case Blue",
    "Siege of Leningrad",
    "Operation Blue",
    "Operation Uranus",
    "German invasion of Poland",
    "Battle of Kalach",
    "Kalach-on-Don",
    "Jassy-Kishinev Offensive",
    "Caucaus Campaign",
    "Caucaus oil fields",
    "Baku",
    "Generalplan Ost",
    "Friedrich Paulus",
    "Erwin Rommel",
    "Erich von Manstein",
    "Georgy Zhukov",
    "Konstantin Rokossovsky",
    "Heinz Guderian",
    # --- KEYWORDS ADICIONALES ---
    "Military logistics of Nazi Germany",
    "Russian railway gauge",
    "Oil campaign of World War II",
    "Blitzkrieg",
    "Two-front war",
    "Barbarossa decree",
    "American entry into World War II",
    "Hypothetical Axis victory in World War II",
    "Axis strategy",
    "Lend-Lease",
    "Economy of Nazi Germany",
    "Military production during World War II",
    "German anti-partisan operations in World War II",
    "Soviet Union in World War II",
    "Wehrmacht",
    "Caucasus oil strategy",
    "Battle of Stalingrad's impacts on civilians",
    "6th Army (Wehrmacht)",
    "Soviet-Japanese Neutrality Pact",
    "Hokushin-ron",
    "Nanshin-ron",
    "Khalkhin Gol",
    "Aerial warfare during Operation Barbarossa",
    "Führer Directive 21",
    "Hitler directives",
    "Kalach bridge",
    "Blue Division",
    "Phoney War",
    "Fall Weiss",
    "Combined Operations Headquarters",
    "Fall Gelb",
    "Manstein plan",
    "Battle of France",
    "Balkans campaign WWII",
    "Political views of Adolf Hitler",
    "Alexander von Hartmann",
    "Order No. 227",
    "Barrier troops",
    "Operation Citadel",
    "German strategy 1943",
    "Soviet propaganda",
    "Appeasement",
    "Unthinkable Operation",
    "Special Operations Executive",
    "Battle of the Bzura",
    "Battle of Szack",
    "Battle of Shatsk",
    "Deutsche Reichsbahn",
    "Minsk agreements",
    "Minsk II",
    "Japan and Barbarossa",
    "Panther tank",
    "German tanks in World War II",
    "Vasily Zaitsev",
    "Soviet snipers",
    "Volga",
    "Stalingrad winter",
    "Operation Winter Storm",
    "Focke-Wulf Fw 190",
    "Focke-Wulf Fw 190 operational history",
    "Bombing of Stalingrad",
    "Battle of Stalingrad's impacts on civilians",
    "Association of German National Jews",
    "Einsatzgruppen",
    "Wannsee Conference",
    "Siege of Leningrad",
    "Mikhail Shumilov",
    "Auschwitz concentration camp",
    "Auschwitz-Birkenau",
    "Treblinka extermination camp",
    "Sobibor extermination camp",
    "Belzec extermination camp",
    "Majdanek concentration camp",
    "Chelmno extermination camp",
    "Holocaust victims",
    "Death toll of Auschwitz",
    "Bombing of Dresden",
    "Bombing of Hamburg",
    "Warsaw Ghetto",
    "Warsaw Uprising",
    "German occupation of Poland",
    "Siege of Leningrad civilian casualties",
    "Battle of Stalingrad civilian casualties",
    "Munich Agreement",
    "Balkan Campaign",
    "Caucasus Campaign",
    "Operation Avalanche",
    "Operation Cobra",
    "Operation Goodwood",
    "Operation Dragoon",
    "Operation Anvil",
    "Operation Spring Awakening",
    "Operation Neptune",
]

def fetch_wiki_page(title: str, lang: str = "en") -> dict | None:
    """
    Descarga una página de Wikipedia en texto plano (extract),
    resolviendo redirecciones. Devuelve un dict listo para usar en RAG.
    """
    url = f"https://{lang}.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,      # texto sin HTML
        "redirects": 1,        # seguir redirecciones
        "titles": title,
    }

    # >>> AQUI VA EL HEADERS <<<
    resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None

    page = next(iter(pages.values()))

    if "missing" in page:
        print(f"[WARN] Página no encontrada para: {title}")
        return None

    extract = (page.get("extract") or "").strip()
    if not extract:
        print(f"[WARN] Página sin extracto para: {title}")
        return None

    normalized_title = page.get("title", title)
    pageid = page.get("pageid")

    doc_id = f"wiki_{pageid}" if pageid is not None else f"wiki_{normalized_title.replace(' ', '_')}"

    doc = {
        "id": doc_id,
        "texto": extract,
        "fuente": "wikipedia",
        "metadata": {
            "title": normalized_title,
            "lang": lang,
            "pageid": pageid,
            "url": f"https://{lang}.wikipedia.org/?curid={pageid}" if pageid is not None else None,
            "original_query": title,
        },
    }
    return doc

def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / "wiki_docs.jsonl"

    docs_guardados = 0

    with open(out_path, "w", encoding="utf-8") as f_out:
        for kw in KEYWORDS:
            try:
                print(f"[INFO] Descargando: {kw} ...")
                doc = fetch_wiki_page(kw, lang="en")
                if doc is None:
                    continue

                f_out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                docs_guardados += 1

                time.sleep(0.5)  # pequeña pausa

            except requests.HTTPError as e:
                print(f"[HTTP ERROR] '{kw}': {e}")
            except Exception as e:
                print(f"[ERROR] Problema con '{kw}': {e}")

    print(f"[DONE] Documentos de Wikipedia guardados en: {out_path}")
    print(f"[DONE] Total de documentos guardados: {docs_guardados}")

if __name__ == "__main__":
    main()