#!/usr/bin/env python3
"""Linguateka: reproducible 3-ASR / 4-judge benchmark.

One download per clip, each STT called once, four text judges over cached STT output.
Reference text is used ONLY after model outputs are final. No interactive Python UI.
Run with: python benchmark.py --dataset dataset.json --out results
Offline:   python benchmark.py --check
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import unicodedata

import requests

CONFIGS = {
    'openai': ('openai',),
    'assemblyai': ('assemblyai',),
    'elevenlabs': ('elevenlabs',),
    'openai+assemblyai+judge': ('openai', 'assemblyai'),
    'openai+elevenlabs+judge': ('openai', 'elevenlabs'),
    'assemblyai+elevenlabs+judge': ('assemblyai', 'elevenlabs'),
    'openai+assemblyai+elevenlabs+judge': ('openai', 'assemblyai', 'elevenlabs'),
}
ASR_MODELS = {'openai': 'gpt-transcribe', 'assemblyai': 'universal-2', 'elevenlabs': 'scribe_v3'}
# Public reference prices, in USD. ESTIMATES ONLY, not invoices.
STT_USD_PER_HOUR = {'openai': 0.0045*60, 'assemblyai': 0.15+0.02, 'elevenlabs': 0.20}
JUDGE_INPUT_USD_PER_M = float(os.getenv('JUDGE_INPUT_USD_PER_M', '0.40'))
JUDGE_OUTPUT_USD_PER_M = float(os.getenv('JUDGE_OUTPUT_USD_PER_M', '1.60'))
JUDGE_MODEL = os.getenv('JUDGE_MODEL', 'gpt-4.1-mini')
LANGUAGE = os.getenv('TRANSCRIPTION_LANGUAGE', 'pl')
MAX_SPEND_USD = float(os.getenv('MAX_ESTIMATED_USD', '10.00'))
HTTP = requests.Session()
LOCK = threading.RLock()
LOG = logging.getLogger('linguateka')


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def valid_secret(name):
    val = os.getenv(name, '')
    if not val or val.startswith('ta_cle'):
        raise RuntimeError(f'Variable absente : {name}')
    return val


def safe_error(err):
    text = str(err)
    for name in ('OPENAI_API_KEY','ASSEMBLYAI_API_KEY','ELEVENLABS_API_KEY'):
        secret = os.getenv(name, '')
        if secret:
            text = text.replace(secret, '[REDACTED]')
    return text[:2200]


def api_call(method, url, *, max_tries=3, **kwargs):
    timeout = kwargs.pop('timeout', (20, 180))
    for attempt in range(max_tries):
        try:
            r = requests.request(method, url, timeout=timeout, **kwargs)
        except (requests.Timeout, requests.ConnectionError):
            if attempt + 1 >= max_tries:
                raise
            time.sleep(2**attempt + 1)
            continue
        if r.status_code in (429, 500, 502, 503, 504) and attempt + 1 < max_tries:
            delay = r.headers.get('Retry-After')
            try:
                delay = min(25, float(delay)) if delay else 2**attempt+1
            except ValueError:
                delay = 2**attempt+1
            time.sleep(delay)
            continue
        if not r.ok:
            raise RuntimeError(f'HTTP {r.status_code} via {url.split("?")[0]}: {r.text[:950]}')
        return r
    raise RuntimeError('Nombre maximal de tentatives atteint')


def shell(cmd, *, timeout=180):
    p=subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(f'Commande échouée ({p.returncode}): {cmd[0]}: {(p.stderr or p.stdout)[-1200:]}')
    return (p.stdout or '').strip()


def download_and_extract(url, tmpdir):
    path=Path(tmpdir)
    out_template=str(path/'source.%(ext)s')
    t0=time.perf_counter()
    shell(['yt-dlp','--no-playlist','--no-warnings','--retries','2','-f','mp4/best',
           '--output',out_template,url],timeout=165)
    sources=[p for p in path.glob('source.*') if p.is_file()]
    if not sources:
        raise RuntimeError('yt-dlp n\'a pas produit de fichier')
    source=sources[0]
    download_s=time.perf_counter()-t0
    audio=path/'audio.mp3'
    t1=time.perf_counter()
    shell(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(source),
           '-map','0:a:0','-vn','-ac','1','-ar','16000','-codec:a','libmp3lame','-b:a','96k',str(audio)],timeout=180)
    duration=float(shell(['ffprobe','-v','error','-show_entries','format=duration',
                           '-of','default=noprint_wrappers=1:nokey=1',str(audio)]))
    if duration<=0 or audio.stat().st_size<1000:
        raise RuntimeError('Fichier audio non valide')
    return audio, duration, download_s, time.perf_counter()-t1


def transcribe_openai(audio):
    from openai import OpenAI
    client=OpenAI(api_key=valid_secret('OPENAI_API_KEY'),timeout=180,max_retries=2)
    t0=time.perf_counter()
    with open(audio,'rb') as f:
        tr=client.audio.transcriptions.create(file=f, model=ASR_MODELS['openai'],
                   response_format='json',languages=[LANGUAGE])
    duration=time.perf_counter()-t0
    txt=(tr.text or '').strip()
    if not txt: raise RuntimeError('OpenAI : texte vide')
    return {'text':txt,'turns':[], 'words':[], 'provider_payload':tr.model_dump(exclude_none=True),
            'model_used':ASR_MODELS['openai'],'latency_s':duration}


def transcribe_assemblyai(audio):
    key=valid_secret('ASSEMBLYAI_API_KEY')
    headers={'authorization':key}
    root='https://api.assemblyai.com/v2'
    t0=time.perf_counter()
    # Bytes: a retry must resend the FULL MP3, not an exhausted file stream.
    payload_bytes=Path(audio).read_bytes()
    r=api_call('POST',root+'/upload',headers=headers,data=payload_bytes,timeout=(15,120))
    upload_url=r.json()['upload_url']
    r=api_call('POST',root+'/transcript',headers=headers,json={
        'audio_url':upload_url,'speech_models':['universal-2'],
        'language_code':LANGUAGE,'speaker_labels':True
    },timeout=(20,60))
    tid=r.json()['id']
    deadline=time.monotonic()+360
    while time.monotonic()<deadline:
        r=api_call('GET',root+'/transcript/'+tid,headers=headers,timeout=(20,60))
        payload=r.json()
        status=payload.get('status')
        if status=='completed': break
        if status=='error': raise RuntimeError(f'AssemblyAI job {tid}: {payload.get("error")}')
        time.sleep(2)
    else:
        raise RuntimeError(f'AssemblyAI polling >360s (job id={tid})')
    text=(payload.get('text') or '').strip()
    if not text: raise RuntimeError('AssemblyAI : texte vide')
    turns=[]; words=[]
    for seg in (payload.get('utterances') or []):
        label=seg.get('speaker') or '?'
        if isinstance(label,int): label=chr(65+label) if 0<=label<26 else '?'
        turns.append({'speaker':str(label),'text':seg.get('text',''),
                      'start':seg.get('start',0)/1000,'end':seg.get('end',0)/1000})
    for word in (payload.get('words') or []):
        label=word.get('speaker') or '?'
        if isinstance(label,int): label=chr(65+label) if 0<=label<26 else '?'
        words.append({'speaker':str(label),'text':word.get('text',''),
                      'start':word.get('start',0)/1000,'end':word.get('end',0)/1000})
    return {'text':text,'turns':turns,'words':words,'provider_payload':payload,
            'model_used':payload.get('speech_model_used') or ASR_MODELS['assemblyai'],
            'latency_s':time.perf_counter()-t0}


def transcribe_elevenlabs(audio):
    key=valid_secret('ELEVENLABS_API_KEY')
    t0=time.perf_counter()
    payload_bytes=Path(audio).read_bytes()
    r=api_call('POST','https://api.elevenlabs.io/v1/speech-to-text',
         headers={'xi-api-key':key},
         data={'model_id':'scribe_v3','language_code':LANGUAGE,
               'diarize':'true','timestamps_granularity':'word',
               'tag_audio_events':'false','no_verbatim':'false'},
         files={'file':(audio.name,payload_bytes,'audio/mpeg')},timeout=(20,240))
    payload=r.json()
    text=(payload.get('text') or '').strip()
    if not text: raise RuntimeError('ElevenLabs : texte vide')
    tokens=payload.get('words') or []
    id_to_label={}
    words=[]; turns=[]
    pieces=[]; old_sp=None; cur_start=None; cur_end=None
    def label(id):
        if id is None or id=='unknown': return '?'
        if id not in id_to_label:
            index=len(id_to_label)
            id_to_label[id]=chr(65+index) if index<26 else f'S{index+1}'
        return id_to_label[id]
    def finish():
        nonlocal pieces, cur_start, cur_end
        value=''.join(pieces).strip()
        if value:
            turns.append({'speaker':old_sp or '?','text':value,'start':cur_start,'end':cur_end})
        pieces=[];cur_start=None;cur_end=None
    for token in tokens:
        typ=token.get('type')
        if typ=='spacing':
            if pieces: pieces.append(token.get('text',' '))
            continue
        if typ not in ('word',None): continue
        text_word=token.get('text','')
        if not text_word: continue
        sp=label(token.get('speaker_id'))
        if old_sp is not None and sp != old_sp:
            finish()
        old_sp=sp
        if pieces and not pieces[-1].endswith((' ','\n','\t')) and not text_word.startswith(('.',',','!','?',':',';')):
            pieces.append(' ')
        pieces.append(text_word)
        st=token.get('start');en=token.get('end')
        if isinstance(st,(int,float)) and isinstance(en,(int,float)):
            if cur_start is None:cur_start=st
            cur_end=en
        words.append({'speaker':sp,'text':text_word,'start':st,'end':en})
    finish()
    return {'text':text,'turns':turns,'words':words,'provider_payload':payload,
            'model_used':'scribe_v3','latency_s':time.perf_counter()-t0}


JUDGE_SYSTEM = '''You are an expert independent ASR transcript adjudicator for Polish audio.
You are given multiple ALTERNATIVE MACHINE transcripts of ONE clip. You do NOT receive audio.
Your task is to produce the best possible VERBATIM Polish transcription by choosing the most probable
word or phrase using grammar, dialogue context, speaker turns, and independent model agreement.
No reference transcript is provided. Do not invent words merely to make language natural.
Never translate. Preserve slang, mistakes made by speakers, disfluencies, profanities, names.
Preserve all utterances; never omit an interjection just because one ASR omitted it.
If uncertain, prefer a word attested in the inputs. Keep timestamps out of text.
Use the speaker labels/turn boundaries of the specified diarization ANCHOR as a starting point.
If there is strong evidence they are wrong, you may correct them; never invent extra speakers casually.
Do not use the same confidence for all corrections. Speaker labels may differ between ASR vendors.
Output only a JSON object with keys:
  "turns": [{"speaker": "A", "text": "..."}, ...],
  "uncertainties": [{"excerpt": "...", "alternatives": ["...", "..."], "reason": "..."}],
  "notes": "one short sentence".
For one-speaker clips, just one A turn is fine. Avoid empty turns. No markdown or prose around JSON.'''


def judge_transcripts(models, outputs):
    from openai import OpenAI
    client=OpenAI(api_key=valid_secret('OPENAI_API_KEY'),timeout=180,max_retries=2)
    # Transcript labels from the anchor, NOT the reference document.
    anchor='elevenlabs' if 'elevenlabs' in models else 'assemblyai'
    entries=[]
    for model in models:
        result=outputs[model]
        if result.get('turns'):
            body='\n'.join(f"{t['speaker']}: {t['text']}" for t in result['turns'])
        else:
            body=result['text']
        entries.append({'model':ASR_MODELS[model],'provider':model,'transcription':body})
    user={'language':LANGUAGE,'diarization_anchor':anchor,'transcriptions':entries,
          'note':'No audio and no verified reference are available. If an uncertainty cannot be settled, list it.'}
    t0=time.perf_counter()
    response=client.chat.completions.create(model=JUDGE_MODEL,temperature=0,
         response_format={'type':'json_object'},max_completion_tokens=6500,
         messages=[{'role':'system','content':JUDGE_SYSTEM},
                   {'role':'user','content':json.dumps(user,ensure_ascii=False)}])
    latency=time.perf_counter()-t0
    raw=response.choices[0].message.content or ''
    parsed=json.loads(raw)
    turns=[]
    for t in parsed.get('turns',[]):
        if not isinstance(t,dict):continue
        val=str(t.get('text','')).strip()
        if val:
            turns.append({'speaker':str(t.get('speaker') or '?').strip(), 'text':val})
    if not turns:
        raise RuntimeError('Le juge n\'a fourni aucune réplique')
    usage=response.usage
    p=int(usage.prompt_tokens if usage else 0)
    c=int(usage.completion_tokens if usage else 0)
    judge_usd=(p*JUDGE_INPUT_USD_PER_M+c*JUDGE_OUTPUT_USD_PER_M)/1_000_000
    return {'text':'\n'.join(t['text'] for t in turns),'turns':turns,
            'uncertainties':parsed.get('uncertainties',[]),
            'judge_notes':parsed.get('notes',''),
            'judge_usage':{'input_tokens':p,'output_tokens':c,'estimated_usd':judge_usd},
            'judge_latency_s':latency,'judge_model':JUDGE_MODEL,
            'judge_response_id':response.id}


def normalize_text(text):
    text=unicodedata.normalize('NFKC',str(text)).lower()
    # Label A:, -B:, or speaker_1: at beginning of a line, NOT ordinary words.
    text=re.sub(r'(?m)^\s*-?\s*(?:[a-z]|s\d+)\s*:\s*','',text)
    # Numbered text lines in source document are list labels (Duolingo clips).
    text=re.sub(r'(?m)^\s*\d{1,2}[.)]\s+','',text)
    return re.findall(r'[^\W_]+(?:[’\x27][^\W_]+)*',text,flags=re.UNICODE)


def ed_stats(reference, hypothesis):
    a=normalize_text(reference);b=normalize_text(hypothesis)
    n=len(a);m=len(b)
    # DP edit distance with correct substitution/deletion/insertion breakdown.
    prev=[(j,0,0,j) for j in range(m+1)]
    for i in range(1,n+1):
        cur=[(i,0,i,0)]
        for j in range(1,m+1):
            if a[i-1]==b[j-1]:
                diag=prev[j-1]
            else:
                diag=tuple(x+y for x,y in zip(prev[j-1],(1,1,0,0)))
            dele=tuple(x+y for x,y in zip(prev[j],(1,0,1,0)))
            ins=tuple(x+y for x,y in zip(cur[-1],(1,0,0,1)))
            cur.append(min(diag,dele,ins,key=lambda c:(c[0],c[1],c[3])))
        prev=cur
    errs,subs,dels,ins=prev[-1]
    # CER, with punctuation/whitespace ignored, accents preserved.
    ca=''.join(a);cb=''.join(b)
    # Levenshtein char edit distance, memory efficient.
    d=list(range(len(cb)+1))
    for i,ch in enumerate(ca,1):
        new=[i]
        for j,bch in enumerate(cb,1):
            new.append(min(d[j]+1,new[-1]+1,d[j-1]+int(ch!=bch)))
        d=new
    return {'reference_words':n,'prediction_words':m,'substitutions':subs,
            'deletions':dels,'insertions':ins,'word_errors':errs,
            'wer':errs/n if n else None,'cer':d[-1]/len(ca) if ca else None,
            'exact_match':int(a==b)}


def token_speakers(turns):
    pairs=[]
    for t in turns or []:
        sp=str(t.get('speaker') or '?').upper().strip()
        pairs.extend((token,sp) for token in normalize_text(t.get('text','')))
    return pairs


def diarization_agreement(reference, predicted):
    """Word-aligned rough accuracy, not DER; only labeled reference lines are scored."""
    truth=[]
    for line in reference.splitlines():
        m=re.match(r'^\s*-?\s*([A-Z])\s*:\s*(.*)',line.strip())
        if m:
            truth.extend((word,m.group(1)) for word in normalize_text(m.group(2)))
        elif line.strip():
            # Untagged continuation line -> cannot safely infer a speaker.
            # We simply omit it from speaker analysis.
            pass
    hyp=token_speakers(predicted)
    if not truth or not hyp:return None
    a=[w for w,_ in truth];b=[w for w,_ in hyp]
    matches=[]
    for tag,i,j,k,l in SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
        if tag=='equal':
            matches.extend((truth[x][1],hyp[y][1]) for x,y in zip(range(i,j),range(k,l)))
    if not matches:return None
    # Speaker labels are arbitrary, optimize mapping pred->truth (small sets).
    from itertools import permutations
    actual=sorted({v for v,_ in matches});pred=sorted({v for _,v in matches})
    if len(pred)>7 or len(actual)>7:
        return {'speaker_accuracy_aligned':None,'speaker_matched_words':len(matches)}
    candidates=actual+['!unmapped']*(max(0,len(pred)-len(actual)))
    best=0
    for perm in set(permutations(candidates,len(pred))):
        mapping=dict(zip(pred,perm))
        score=sum(mapping[b]==a for a,b in matches)
        best=max(best,score)
    return {'speaker_accuracy_aligned':best/len(matches),
            'speaker_matched_words':len(matches),'speaker_reference_words':len(truth)}


def path_for(base, video_id, provider):
    return base/'cache'/video_id/f'{provider}.json'


def save_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
    tmp.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def append_event(out,event,**details):
    row={'ts_utc':utcnow(),'event':event,**details}
    with LOCK:
        with open(out/'events.jsonl','a',encoding='utf-8') as f:
            f.write(json.dumps(row,ensure_ascii=False,default=str)+'\n')
    LOG.info('%s %s',event, {k:v for k,v in details.items() if k not in ('reference_text',)})


def run_video(video,out, resume=True):
    vid=video['id'];reference=video['reference_text'];url=video.get('url')
    save_json(out/'references'/f'{vid}.json',video)
    events={'id':vid,'title':video['title'],'url':url,'date_utc':utcnow(),
            'source_reference':reference,'asr':{},'configurations':{},'errors':[]}
    cached_asr={}
    for p in ASR_MODELS:
        file=path_for(out,vid,p)
        if resume and file.exists():
            try:
                cached_asr[p]=load_json(file)
            except Exception: pass
    incomplete=set(ASR_MODELS)-set(cached_asr)
    with tempfile.TemporaryDirectory(prefix='linguateka_') as tmp:
        if incomplete:
            try:
                if not url: raise RuntimeError('URL manquante dans dataset')
                append_event(out,'download_start',video=vid,url=url)
                audio,duration,dl_s,ex_s=download_and_extract(url,tmp)
                events['download']={'duration_audio_s':duration,'download_s':dl_s,'extract_s':ex_s}
                append_event(out,'download_ok',video=vid,duration_s=duration,
                             download_s=round(dl_s,2),extract_s=round(ex_s,2))
            except Exception as e:
                err=safe_error(e)
                append_event(out,'download_fail',video=vid,error=err)
                events['errors'].append({'stage':'download','error':err,'traceback':traceback.format_exc()[-3000:]})
                save_json(out/'videos'/f'{vid}.json',events)
                return events
            # All three transcription engines start from the SAME decoded audio.
            methods={'openai':transcribe_openai,'assemblyai':transcribe_assemblyai,'elevenlabs':transcribe_elevenlabs}
            with ThreadPoolExecutor(max_workers=len(incomplete)) as pool:
                jobs={pool.submit(methods[p],audio):p for p in incomplete}
                for fut in as_completed(jobs):
                    p=jobs[fut]
                    try:
                        item=fut.result()
                        item['estimated_usd']=duration/3600*STT_USD_PER_HOUR[p]
                        item['audio_duration_s']=duration
                        cached_asr[p]=item
                        save_json(path_for(out,vid,p),item)
                        append_event(out,'asr_ok',video=vid,provider=p,
                          latency_s=round(item['latency_s'],3),model=item['model_used'],
                          estimated_usd=round(item['estimated_usd'],6))
                    except Exception as e:
                        error=safe_error(e)
                        append_event(out,'asr_fail',video=vid,provider=p,error=error)
                        events['errors'].append({'stage':p,'error':error,'traceback':traceback.format_exc()[-3000:]})
    # Cached outputs reused across ALL hybrid configurations.
    for p,item in cached_asr.items():
        events['asr'][p]=item
    for cfg,providers in CONFIGS.items():
        missing=[p for p in providers if p not in cached_asr]
        if missing:
            append_event(out,'config_skipped',video=vid,config=cfg,missing=missing)
            continue
        if len(providers)==1:
            p=providers[0];a=cached_asr[p]
            result={'text':a['text'],'turns':a.get('turns',[]),'judge_usage':None,
                    'judge_latency_s':0.0,'source_models':list(providers),
                    'estimated_cost_usd':a.get('estimated_usd',0),
                    'parallel_latency_s':a.get('latency_s',0)}
        else:
            cached_path=out/'cache'/vid/f'{cfg.replace("+","_")}.json'
            try:
                if resume and cached_path.exists():
                    result=load_json(cached_path)
                else:
                    append_event(out,'judge_start',video=vid,config=cfg,judge=JUDGE_MODEL)
                    result=judge_transcripts(providers,cached_asr)
                    result['source_models']=list(providers)
                    result['estimated_cost_usd']=(sum(cached_asr[p].get('estimated_usd',0) for p in providers)
                                                    +result['judge_usage']['estimated_usd'])
                    result['parallel_latency_s']=(max(cached_asr[p].get('latency_s',0) for p in providers)
                                                  +result['judge_latency_s'])
                    save_json(cached_path,result)
                    append_event(out,'judge_ok',video=vid,config=cfg,
                      latency_s=round(result['judge_latency_s'],3),
                      tokens=result['judge_usage'],estimated_usd=round(result['estimated_cost_usd'],6))
            except Exception as e:
                error=safe_error(e)
                append_event(out,'judge_fail',video=vid,config=cfg,error=error)
                events['errors'].append({'stage':cfg,'error':error,'traceback':traceback.format_exc()[-3000:]})
                continue
        result['metrics']=ed_stats(reference,result['text'])
        diar=diarization_agreement(reference,result.get('turns',[]))
        if diar:result['metrics'].update(diar)
        events['configurations'][cfg]=result
        append_event(out,'score',video=vid,config=cfg,
                     wer=round(result['metrics']['wer'],4),
                     cer=round(result['metrics']['cer'],4),
                     estimated_usd=round(result['estimated_cost_usd'],6))
    events['finished_at_utc']=utcnow()
    save_json(out/'videos'/f'{vid}.json',events)
    return events


def check(dataset):
    entries=dataset['videos']
    print(f'Dataset : {len(entries)} entrées | {len(set(x.get("url") for x in entries))} URLs uniques | {len(set(x["id"] for x in entries))} IDs')
    for video in entries:
        warnings=video.get('warnings') or []
        if not video.get('reference_text') or not video.get('url') or warnings:
            print('ATTENTION:',video['id'],video['title'],warnings)
    if not entries:
        raise RuntimeError('dataset vide')
    if len(entries)!=len(set(x.get('url') for x in entries)):
        raise RuntimeError('URLs dupliquées')
    if len(entries)!=len(set(x['id'] for x in entries)):
        raise RuntimeError('IDs dupliqués')
    print('OK : textes et liens présents. Accessibilité Instagram non vérifiée (téléchargement à l’exécution).')


def main():
    ap=argparse.ArgumentParser(description='Benchmark Linguateka multi-ASR / juge contextuel')
    ap.add_argument('--dataset',default='dataset.json')
    ap.add_argument('--out',default='results')
    ap.add_argument('--limit',type=int,default=0,help='0 = toutes les vidéos')
    ap.add_argument('--ids',default='',help='IDs séparés par virgule, ex. 001,003,016b')
    ap.add_argument('--check',action='store_true',help='Vérifier seulement les liens dans le fichier et la structure')
    ap.add_argument('--fresh',action='store_true',help='Forcer réexécution des appels payants (déconseillé)')
    args=ap.parse_args()
    data=load_json(args.dataset)
    check(data)
    if args.check:
        return
    # Fail before any billable request if one required secret missing.
    for key in ('OPENAI_API_KEY','ASSEMBLYAI_API_KEY','ELEVENLABS_API_KEY'):
        valid_secret(key)
    for prog in ('yt-dlp','ffmpeg','ffprobe'):
        if not shutil.which(prog): raise RuntimeError(f'Exécutable introuvable : {prog}')
    out=Path(args.out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    loghandler=logging.FileHandler(out/'benchmark.log',encoding='utf-8')
    loghandler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    LOG.addHandler(loghandler)
    LOG.setLevel(logging.INFO)
    vids=data['videos']
    if args.ids:
        wanted={s.strip() for s in args.ids.split(',') if s.strip()}
        vids=[v for v in vids if v['id'] in wanted]
    if args.limit>0:vids=vids[:args.limit]
    prices={'stt_usd_per_hour':STT_USD_PER_HOUR,'judge_input_usd_per_million_tokens':JUDGE_INPUT_USD_PER_M,
            'judge_output_usd_per_million_tokens':JUDGE_OUTPUT_USD_PER_M}
    save_json(out/'run_config.json',{'started_utc':utcnow(),'dataset':args.dataset,'count':len(vids),
              'models':ASR_MODELS,'judge_model':JUDGE_MODEL,'prices_are_estimates':True,
              'pricing':prices,'max_estimated_usd':MAX_SPEND_USD})
    append_event(out,'run_start',videos=len(vids),judge=JUDGE_MODEL)
    for i,video in enumerate(vids,1):
        print(f'\n[{i}/{len(vids)}] {video["id"]} {video["title"]}',flush=True)
        # Budget guard before each video, based on successful results recorded to disk.
        completed=list((out/'videos').glob('*.json')) if (out/'videos').exists() else []
        # Actual estimated total = unique STT charges + all JUDGE charges; don't sum configurations.
        asr_cost=0.0;judge_cost=0.0
        for path in (out/'cache').glob('**/*.json') if (out/'cache').exists() else []:
            try:
                item=load_json(path)
                if path.stem in ASR_MODELS:
                    asr_cost+=float(item.get('estimated_usd',0))
                elif item.get('judge_usage'):
                    judge_cost+=float(item['judge_usage']['estimated_usd'])
            except (ValueError,KeyError,TypeError):pass
        if asr_cost+judge_cost >= MAX_SPEND_USD:
            append_event(out,'budget_stop',spent_estimated_usd=asr_cost+judge_cost,
                         max_estimated_usd=MAX_SPEND_USD)
            print('STOP : plafond de dépenses estimées atteint',flush=True)
            break
        try:
            run_video(video,out,resume=not args.fresh)
        except Exception as e:
            append_event(out,'video_unhandled_error',video=video['id'],error=safe_error(e),
                         traceback=traceback.format_exc()[-4000:])
    append_event(out,'run_finish')
    try:
        from report import build_report
        build_report(out)
    except Exception as e:
        append_event(out,'report_failed',error=safe_error(e),traceback=traceback.format_exc()[-4000:])
        raise
    print('\nTERMINE : rapports, CSV, JSON et graphiques enregistrés dans',out,flush=True)

if __name__=='__main__':
    main()
