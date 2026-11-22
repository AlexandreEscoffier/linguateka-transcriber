# Linguateka — banc d'essai de 37 Reels en polonais

Un programme **sans interface graphique** qui télécharge les vidéos Instagram indiquées dans `dataset.json`, compare trois moteurs de transcription audio puis teste quatre arbitrages contextuels. Les **17 graphiques PNG**, les fichiers CSV, les JSON bruts et tous les journaux sont construits automatiquement. Les **références ne sont jamais envoyées aux modèles**.

## Pourquoi 37 vidéos et non 35 ?

Le Word envoyé contenait **36 vidéos numérotées et « Duolinguo dans la vraie vie » sans numéro**. Les **37 URL Instagram ont été retrouvées dans les hyperliens Word**, puis enregistrées dans `dataset.json`, avec leurs références textuelles originales. L'extraction est vérifiée hors ligne ; **le téléchargement réel des contenus Instagram n'a pas été confirmé**. S'ils sont privés, supprimés ou bloqués pour les adresses IP GitHub, les erreurs figureront dans `errors.csv` et `events.jsonl`.

## Démarrage : un bouton GitHub Actions

Le plus sûr est d'exécuter les benchmarks dans **GitHub Actions**, *pas dans les requêtes HTTP de Railway*. Ainsi, aucun timeout de navigateur/Laravel ou déploiement Railway ne coupera les tests. **Ton `app.py` actuel n'est pas modifié.**

1. Copie **le contenu de cette archive à la racine** de ton dépôt `linguateka-transcriber` (pas dans un sous-dossier). Ne remplace pas ton fichier `app.py` ni ton `requirements.txt` existants. Les dépendances du banc d'essai sont volontairement dans `benchmark_requirements.txt`.
2. Dans GitHub, ouvre **Settings → Secrets and variables → Actions → New repository secret**. Crée **trois secrets**, avec exactement ces noms : `OPENAI_API_KEY`, `ASSEMBLYAI_API_KEY`, `ELEVENLABS_API_KEY`. Réutilise les valeurs secrètes déjà enregistrées dans Railway, sans les pousser dans Git. La clé OpenAI sert à la transcription **et** au modèle arbitre.
3. Pousse ces fichiers sur la **branche par défaut** du dépôt :

   ```bash
   git add benchmark.py report.py extract_reference.py dataset.json benchmark_requirements.txt tests .github/workflows/benchmark.yml README.md .gitignore
   git commit -m "Add Linguateka 37-video ASR benchmark"
   git push origin master
   ```

   **Si ta branche par défaut est `main`, remplace `master` par `main`.** Si `.gitignore` existe déjà, fusionne son contenu au lieu de le remplacer.
4. Dans GitHub, ouvre l'onglet **Actions** → **« Linguateka - Benchmark ASR (37 videos) »** → **Run workflow**.
5. Pour un premier essai, mets `videos=2` puis clique sur le bouton vert **Run workflow**. Pour le benchmark complet, relance avec `videos=0` (valeur par défaut, les 37 vidéos).
6. Une fois le workflow terminé, ouvre son exécution → section **Artifacts** → télécharge `linguateka-benchmark-results-…`. Décompresse le ZIP : il contient les rapports et les graphiques, que tu peux ensuite joindre ici pour une analyse détaillée.

**Attention :** un nouveau lancement GitHub Actions refacture les appels API. Les résultats sont mis en cache **à l'intérieur d'une exécution**, mais le cache n'est pas conservé automatiquement d'une exécution GitHub à l'autre. Le test avec deux vidéos facture donc ces deux vidéos une nouvelle fois lorsqu'elles seront incluses dans le test des 37.

Si GitHub Actions n'est pas activé sur ton dépôt, il faut l'autoriser dans les paramètres du dépôt. Les secrets sont indispensables même si Railway possède déjà les mêmes variables : **GitHub Actions et Railway ont des environnements différents**.

## Les 7 configurations

| Configuration | Appels API |
|---|---|
| `openai` | GPT-transcribe |
| `assemblyai` | AssemblyAI Universal-2 avec diarisation |
| `elevenlabs` | ElevenLabs Scribe v3 avec diarisation |
| `openai+assemblyai+judge` | GPT + Assembly + GPT-4.1-mini juge |
| `openai+elevenlabs+judge` | GPT + Eleven + juge |
| `assemblyai+elevenlabs+judge` | Assembly + Eleven + juge |
| `openai+assemblyai+elevenlabs+judge` | Les trois + juge |

**Chaque vidéo subit exactement trois appels STT** (un par moteur) et **quatre appels au juge** (les trois paires et le trio). On ne retranscrit donc pas les mêmes fichiers pour chaque paire. Sur 37 vidéos complètes, le maximum prévu est **111 transcriptions + 148 jugements** hors tentatives de reprise sur erreurs temporaires.

Les trois STT commencent en parallèle pour une même vidéo. Les vidéos sont traitées l'une après l'autre. Cela ne mobilise donc pas tes 12 emplacements ElevenLabs simultanément.

## Comment sont réalisés les jugements ?

L'algorithme envoie au juge les **deux ou trois textes candidats et les locuteurs renvoyés par le fournisseur**, jamais les références réelles. `gpt-4.1-mini` doit proposer un texte final en polonais et les répliques A/B/etc. Les changements controversés sont consignés dans `uncertainties`. Il n'écoute pas l'audio : il peut être trompé si toutes les transcriptions sont fausses, et peut introduire de nouvelles erreurs. Tous les textes bruts sont conservés pour audit.

Pour les jugements, l'ancre de diarisation est **ElevenLabs** lorsque présent, sinon **AssemblyAI**. Les étiquettes restent approximatives lorsque les deux modèles attribuent différemment les voix.

## Tarifs utilisés pour les ESTIMATIONS

Tarifs publics vérifiés le 22 novembre 2026, susceptibles de varier selon ton compte :

- `gpt-transcribe` : **0,0045 $/minute**.
- `AssemblyAI Universal-2` : **0,15 $/heure**, + **0,02 $/heure de diarisation**.
- `ElevenLabs Scribe v3 batch` : **0,20 $/heure** pour les comptes concernés.
- `GPT-4.1-mini` juge : **0,40 $ / million de tokens en entrée**, **1,60 $ / million en sortie** (prix de référence, à vérifier selon l'offre).

Ces tarifs sont configurables directement dans `benchmark.py` ou, pour le juge, les variables `JUDGE_INPUT_USD_PER_M` et `JUDGE_OUTPUT_USD_PER_M`. Le code récupère les tokens du juge quand ils sont renvoyés, mais **les coûts affichés sont estimés, pas des factures**. Il ne tient pas compte de taxes, minimums, crédits promotionnels, tarification personnalisée, éventuels appels facturés après timeout ni coût GitHub/Railway.

`MAX_ESTIMATED_USD=10` est une **garde-fou logiciel approximatif**, pas un plafond réel. Mets également des limites de dépenses dans les consoles des trois fournisseurs si tu veux contenir les frais.

## Comment lire les résultats

- `summary.csv` — classement global des 7 configurations : WER pondéré, WER moyen, CER, coût de configuration, latences, accord des locuteurs.
- `per_video.csv` — l'ensemble des textes, erreurs et timings pour chaque Reel et chaque configuration.
- `differences.csv` — substitutions, insertions et suppressions détaillées par passage.
- `errors.csv` — échec de téléchargement, API ou jugement, avec motif.
- `actual_run_costs.csv` — coût total **réellement sollicité en estimation** (3 STT + 4 juges par vidéo, sans recompter les STT partagés).
- `videos/*.json` — données regroupées par vidéo.
- `cache/*/*.json` — données API originales et réponses de juge. **Ne publie pas publiquement les transcriptions ou les données privées.**
- `events.jsonl` et `benchmark.log` — traces chronologiques, réussites et échecs.
- `charts/*.png` — **17 graphiques** auto-générés, notamment WER, CER, temps, coût, carte thermique, comparaisons et gains des juges.

**WER** = `(substitutions + suppressions + insertions) / nombre de mots dans le texte de référence`. Le calcul ignore la ponctuation, la casse et les marqueurs `A:`, mais **respecte les accents polonais**. Pour les vidéos dans lesquelles la référence contient des mots incorrects ou des textes écrits à l'écran, le WER peut surestimer les erreurs réelles des modèles.

**Speaker accuracy** est une **estimation d'accord sur les mots identiques alignés**, après permutation des étiquettes. Ce n'est **pas** un véritable `DER` calculé sur des temps de référence, car ton Word ne contient pas les temps exacts des interventions. OpenAI sans diarisation n'a pas de score locuteurs.

**Latence hybride** = `max(durée réelle des appels STT du groupe) + durée du juge`, une estimation de durée si les moteurs démarrent ensemble. Les timings de chaque appel API sont bien mesurés en conditions réelles.

## Exécution directe (ordinateur ou conteneur Railway)

Si tu veux lancer le script indépendamment de GitHub Actions :

```bash
python -m pip install -r benchmark_requirements.txt
python benchmark.py --check
python -m unittest discover -s tests -v
# Nécessite : yt-dlp, ffmpeg, ffprobe et les trois variables de clé API
python benchmark.py --out results --limit 2
python benchmark.py --out results   # les 37 vidéos
python report.py results          # reconstruire les graphiques sans API
```

Le programme n'a aucun formulaire, aucune page Web et aucun bouton Python. **Le bouton de déclenchement est celui de GitHub Actions.**

## Si Instagram bloque les téléchargements

Si l'artifact contient beaucoup d'événements `download_fail` : Instagram peut refuser les IP des runners GitHub, demander une connexion, ou supprimer certaines publications. Ce n'est pas un problème de clé API STT. Le script ne peut pas contourner magiquement une vidéo privée. Dans ce cas, il faudra exécuter le benchmark sur un environnement disposant d'un accès légitime aux Reels (par exemple ton Railway si ses accès fonctionnent toujours) ou fournir des MP3/MP4 que tu as le droit d'utiliser. **Aucun téléchargement réel Instagram n'a été confirmé pendant la préparation de ce package.**

## Documentation officielle

- OpenAI transcription : https://developers.openai.com/api/docs/guides/speech-to-text
- AssemblyAI diarisation : https://www.assemblyai.com/docs/pre-recorded-audio/label-speakers
- ElevenLabs Scribe v3 API : https://elevenlabs.io/docs/api-reference/speech-to-text/convert
- Prix ElevenLabs (17 novembre 2026) : https://elevenlabs.io/docs/changelog/2026/11/17
