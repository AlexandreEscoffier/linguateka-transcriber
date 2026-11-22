"""Extract original references and Instagram hyperlinks from a Word document.
Usage: python extract_reference.py 'Fichier de référence du polonais.docx' --out dataset.json
"""
import argparse, json, re, sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from docx import Document
from docx.oxml.ns import qn

HEADING = re.compile(r'^Vid[eé]o\s+(\d+)\s+(.+)$', re.I)

def canonical_url(url):
    match = re.match(r'^https?://(?:www\.)?instagram\.com/(p|reel|tv)/([\w-]+)/?', url)
    if not match:
        return None
    return f'https://www.instagram.com/{match.group(1)}/{match.group(2)}/'

def paragraph_links(para, doc):
    values = []
    for h in para._p.xpath('.//w:hyperlink'):
        rid = h.get(qn('r:id'))
        if rid and rid in doc.part.rels:
            target = doc.part.rels[rid].target_ref
            if 'instagram.com/' in target:
                values.append(target)
    return values

def extract(source):
    doc=Document(source)
    clips=[]
    current=None
    def flush():
        if current:
            current['reference_text']='\n'.join(current.pop('paragraphs')).strip()
            if not current['reference_text']:
                current['warnings'].append('Pas de texte de référence')
            if not current.get('url'):
                current['warnings'].append('Lien Instagram absent')
            clips.append(current)
    for p in doc.paragraphs:
        t=p.text.strip()
        if not t and not paragraph_links(p,doc):
            continue
        m=HEADING.match(t)
        if m or t.lower()=='duolinguo dans la vraie vie':
            flush()
            if m:
                num=int(m.group(1)); title=m.group(2).strip(); uid=f'{num:03d}'
            else:
                num=None; title=t; uid='016b'
            current={'id':uid,'number_in_document':num,'title':title,'url':None,'paragraphs':[],'warnings':[]}
            continue
        if current is None:
            continue
        links=paragraph_links(p,doc)
        if links:
            if len(links)>1: current['warnings'].append('Plusieurs liens Instagram dans un même paragraphe')
            url=canonical_url(links[0])
            if not url: current['warnings'].append(f'Lien mal formé : {links[0]}')
            else: current['url']=url
            continue
        if t.lower()=='instagram':
            current['warnings'].append('Texte Instagram sans hyperlien')
            continue
        current['paragraphs'].append(t)
    flush()
    urls=[v['url'] for v in clips if v.get('url')]
    if len(urls)!=len(set(urls)):
        print('ATTENTION : URL en doublon !',file=sys.stderr)
    return {'metadata':{'language':'pl','reference_source':Path(source).name,'clips':len(clips),
              'extracted_at_utc':datetime.now(timezone.utc).isoformat(),
              'notes':'Transcriptions fournies par utilisateur, jamais transmises aux IA. Les références peuvent contenir des erreurs.'},
            'videos':clips}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('docx')
    parser.add_argument('--out',default='dataset.json')
    args=parser.parse_args()
    data=extract(args.docx)
    Path(args.out).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Extrait: {len(data["videos"])} vidéos, {sum(bool(v["url"]) for v in data["videos"])} URL')
    for v in data['videos']:
        print(v['id'],v['title'],v['url'],len(v['reference_text']),v['warnings'])
