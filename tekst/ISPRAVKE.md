# ISPRAVKE master izvještaja (usklađivanje sa kodom)

Dokument nastao unakrsnom provjerom svih 5 poglavlja naspram koda HazopLLM-a.
Za svaku stavku: **lokacija → postojeći tekst → predložena ispravka → referenca u kodu**.
Stavke su sortirane po prioritetu (🔴 visok, 🟡 srednji, 🟢 nizak).

> Napomena: ogroman dio izvještaja je tačan i NE treba ga dirati (11 guidewords + validator,
> L1/L2/L3 modeli, reviewer/repair, holistički review, kaskada, cijeli GUI dio, APA funkcije).
> Ispod su samo stvari koje treba ispraviti.

---

## 🔴 B1 — „Grok" → „Groq"  (poglavlja 4.3.3, 4.3.4, 4.4)

**Problem:** Kod koristi provajdera **`groq`** (Groq Inc., inference platforma; base url
`https://api.groq.com/openai/v1`). U radu piše **„Grok"** — a „Grok" je LLM kompanije xAI,
potpuno drugi model i proizvođač. Ovo je faktička greška koju će komisija lako uočiti.

**Naslovi koje treba izmijeniti:**
- `4.3.3. ... (Grok / Kimi K2)`  →  `4.3.3. ... (Kimi K2, Groq)`
- `4.3.4. ... (Grok / Kimi K2 + RAG)`  →  `4.3.4. ... (Kimi K2, Groq + RAG)`

**U tekstu 4.4** svako „Grok/Kimi" → „Kimi K2 (Groq)" ili samo „Kimi K2".

**Dodatno (važno):** default model za `groq` provajdera u kodu je **Llama**
(`llama-3.1-8b-instant`, review `llama-3.3-70b-versatile`). Kimi K2 je dakle pokrenut preko
`--model` override-a — zato OBAVEZNO navesti tačan korišteni model ID (npr.
`moonshotai/kimi-k2-instruct`) u tabeli konfiguracije (vidi D1).

**Referenca:** `src/llm_client.py:54–57` (PROVIDER_DEFAULTS["groq"]).

---

## 🔴 B2 — TF-IDF nije „semantički embedding" (poglavlje 2.3)

**Problem:** U 2.3 stoji: *„Za razliku od jednostavnih 'bag-of-words' reprezentacija, embedding
modeli čuvaju značenje..."* i kasnije se „lokalni modeli" navode kao embedding modeli. Ali
implementirani **`local`** RAG backend je **TF-IDF (scikit-learn)** — metoda iz porodice
bag-of-words, NE semantički neuronski embedding. Interna protivrječnost.

**Predloženi ispravljeni pasus (zamijeniti dio o lokalnim/cloud embedding modelima u 2.3):**

> Embedding reprezentacije mogu se dobiti na dva načina. Cloud modeli (npr. OpenAI ili Gemini
> embedding modeli) koriste neuronske mreže trenirane na velikim korpusima i daju guste
> (dense) vektore koji hvataju semantičko značenje, pa tekstovi sličnog značenja imaju slične
> vektore i kada ne dijele iste riječi. Kao lokalna, offline alternativa može se koristiti
> klasična leksička reprezentacija poput TF-IDF, koja ne hvata semantiku na isti način, ali ne
> zahtijeva eksterne servise ni slanje podataka. Izbor između leksičkog i semantičkog pristupa
> direktno utiče na kvalitet pretrage i predstavlja kompromis između tačnosti, privatnosti i
> jednostavnosti implementacije.

**Referenca:** `src/rag_utils.py:4–7` (tri backenda: local=TF-IDF, openai, gemini), `:43–44`.

---

## 🟡 C1 — Poglavlje 3.8 (RAG implementacija) dopuniti

**Problem:** 3.8 opisuje SAMO da RAG postoji i koje formate prima, ali ne KAKO radi. Komisija
očekuje implementacione detalje.

**Predloženi dodatni pasus (umetnuti u 3.8, poslije postojećeg teksta o formatima):**

> Implementacija RAG-a funkcioniše na sljedeći način. Učitani dokumenti se dijele na preklapajuće
> isječke (chunks) fiksne veličine (oko 1200 znakova, sa preklapanjem od 200 znakova), čime se
> dugi tekstovi razlažu na jedinice pogodne za pretragu. Svaki isječak se zatim pretvara u
> vektorsku reprezentaciju jednim od tri podržana backenda: lokalni TF-IDF (scikit-learn, offline),
> ili cloud embedding modeli OpenAI odnosno Gemini. Pri generisanju, za svaku fazu se formiraju
> upiti i pretraga se vrši na osnovu kosinusne sličnosti između upita i isječaka, uz vraćanje
> najsličnijih rezultata (top-k, tipično 2–3) iznad zadatog praga sličnosti (parametar
> `rag_min_sim`). Indeks se drži u radnoj memoriji za vrijeme analize i ne koristi se eksterna
> vektorska baza podataka. Pronađeni isječci se prosljeđuju modelu kao dodatni kontekst uz upit
> faze (L1, L2 i L3).

**Reference:** `src/rag_utils.py:164` (chunk_text 1200/200), `:373–392` (cosine + top_k + min_similarity);
`src/graph_full.py:377, 385, 395, 423, 441, 459` (top_k=3 za L1, 2 za L2/L3; min_sim; in-memory rag_mat).

---

## 🟡 C2 — Izmjenjiva polja: dodati `cause` (poglavlje 3.9)

**Postojeći tekst:** *„Korisnik može mijenjati polja deviation, effect i potentially dangerous..."*

**Ispravka:** *„Korisnik može mijenjati polja deviation, **cause**, effect i potentially
dangerous, pri čemu korisničke izmjene imaju prioritet nad originalnim vrijednostima."*

**Referenca:** `src/gui/models.py` (EditRowRequest: deviation, cause, effect, potentially_dangerous);
`src/gui/routes/rows.py:26, 36–39`.

---

## 🟡 C3 — Devijacija po guidewordu (poglavlje 3.1)

**Postojeći tekst:** *„...za svaku kombinaciju funkcije i guideworda formuliše devijaciju..."*
(implicira tačno jednu).

**Ispravka (omekšati):** *„...za svaku kombinaciju funkcije i guideworda formuliše jednu ili
više devijacija (broj je konfigurabilan parametrom `max_devs_per_gw`, podrazumijevano 2), pri
čemu validator zahtijeva da svaki guideword bude zastupljen barem jednom po funkciji."*

**Referenca:** `--max_devs_per_gw` (default 2); `src/validators.py:151` (pokrivenost = „barem jednom").

---

## 🟢 D1 — Tabela konfiguracije / verzije modela (poglavlje 4)

**Problem:** Navode se „GPT-5.2" i „Gemini 2.5 Pro". Default-i u kodu su `gpt-3.5-turbo` i
`gemini-2.5-flash`; Groq default je Llama. Sve korišteno preko `--model` override-a.

**Preporuka:** Dodati u 4.3 (prije pojedinačnih analiza) tabelu eksperimentalne konfiguracije,
npr.:

| Konfiguracija | Provider | Model (generacija) | Model (review) | RAG embedder | max_devs_per_gw |
|---|---|---|---|---|---|
| Gemini 2.5 Pro | gemini | gemini-2.5-pro | gemini-2.5-pro | – / local | 2 |
| Kimi K2 (Groq) | groq | moonshotai/kimi-k2-instruct | … | – / local | 2 |
| GPT-5.2 | openai | gpt-5.2 (provjeriti tačan ID) | … | – / local | 2 |

(Popuniti tačnim ID-jevima koji su STVARNO korišteni — radi reproducibilnosti.)

**Referenca:** `src/llm_client.py:39–58` (PROVIDER_DEFAULTS); CLI `--model`, `--model-review`.

---

## 🟢 D2 — Lektura / numeracija slika (poglavlje 3)

- Human-review dijagram: u tekstu „Na slici **3.6**...", a caption kaže „Slika **3.5**. Dijagram
  human reviewa". Uskladiti na 3.6.
- Tipografske greške (više mjesta): **„sitema" → „sistema"**, **„arhitetkure" → „arhitekture"**.

---

## 🟢 D3 — Opcione funkcionalnosti (nije greška, samo za potpunost)

Mogu se kratko spomenuti ili izostaviti:
- HTML import postojećih rezultata (`src/gui/services/html_importer.py`, ruta `import_html`).
- Polja `applicability_score` / `applicability_note` u L1 redu (`src/models.py`).
- Prag `rag_min_sim` (već pokriven u C1).
