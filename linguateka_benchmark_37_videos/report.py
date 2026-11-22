#!/usr/bin/env python3
"""Offline reporting, no API calls. Re-run: python report.py results"""
from __future__ import annotations
from collections import Counter, defaultdict
import csv
from difflib import SequenceMatcher
import json
import math
from pathlib import Path
import statistics
import sys

from benchmark import CONFIGS, normalize_text


def csv_write(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def diff_fragments(ref,pred):
    a=normalize_text(ref);b=normalize_text(pred)
    diffs=[]
    for tag,i,j,k,l in SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
        if tag!='equal':
            diffs.append({'edit_type':tag,'reference':' '.join(a[i:j]),
                          'hypothesis':' '.join(b[k:l]),'word_index':i})
    return diffs


def build_report(out):
    out=Path(out)
    videos=out/'videos'
    items=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(videos.glob('*.json'))] if videos.exists() else []
    runs=[]; errors=[]; all_diffs=[];unique_cost=[]
    for v in items:
        vid=v['id'];dur=v.get('download',{}).get('duration_audio_s')
        if not dur:
            asrs=list((v.get('asr') or {}).values())
            dur=asrs[0].get('audio_duration_s') if asrs else None
        ASR=v.get('asr',{})
        asr_cost=sum(a.get('estimated_usd',0) for a in ASR.values())
        judge_cost=0.0
        for key,c in v.get('configurations',{}).items():
            metric=c.get('metrics',{})
            judge_info=c.get('judge_usage') or {}
            judge_cost+=judge_info.get('estimated_usd',0)
            record={
                'video_id':vid,'title':v.get('title',''),'config':key,
                'url':v.get('url',''),'audio_seconds':dur,
                'reference_words':metric.get('reference_words'),
                'prediction_words':metric.get('prediction_words'),
                'substitutions':metric.get('substitutions'),
                'deletions':metric.get('deletions'),
                'insertions':metric.get('insertions'),
                'word_errors':metric.get('word_errors'),
                'wer_percent':metric['wer']*100 if metric.get('wer') is not None else None,
                'cer_percent':metric['cer']*100 if metric.get('cer') is not None else None,
                'exact_match':metric.get('exact_match'),
                'speaker_accuracy_aligned_percent':metric['speaker_accuracy_aligned']*100
                    if metric.get('speaker_accuracy_aligned') is not None else None,
                'speaker_matched_words':metric.get('speaker_matched_words'),
                'estimated_cost_usd':c.get('estimated_cost_usd'),
                'parallel_latency_seconds':c.get('parallel_latency_s'),
                'judge_latency_seconds':c.get('judge_latency_s',0),
                'judge_input_tokens':judge_info.get('input_tokens',0),
                'judge_output_tokens':judge_info.get('output_tokens',0),
                'judge_cost_usd':judge_info.get('estimated_usd',0),
                'reference_text':v.get('source_reference',''),
                'prediction_text':c.get('text',''),
                'speaker_turns_json':json.dumps(c.get('turns',[]),ensure_ascii=False),
                'uncertainties_json':json.dumps(c.get('uncertainties',[]),ensure_ascii=False),
            }
            runs.append(record)
            for diff in diff_fragments(v.get('source_reference',''),c.get('text','')):
                all_diffs.append({'video_id':vid,'config':key,'title':v.get('title',''),**diff})
        unique_cost.append({'video_id':vid,'title':v.get('title',''),
                            'api_stt_cost_usd':asr_cost,'judge_total_cost_usd':judge_cost,
                            'estimated_actual_spend_this_run_usd':asr_cost+judge_cost,
                            'audio_seconds':dur})
        for e in v.get('errors',[]):
            errors.append({'video_id':vid,'title':v.get('title'),**e})
    columns=['video_id','title','config','url','audio_seconds','reference_words','prediction_words',
      'substitutions','deletions','insertions','word_errors','wer_percent','cer_percent',
      'exact_match','speaker_accuracy_aligned_percent','speaker_matched_words',
      'estimated_cost_usd','parallel_latency_seconds','judge_latency_seconds',
      'judge_input_tokens','judge_output_tokens','judge_cost_usd','reference_text','prediction_text',
      'speaker_turns_json','uncertainties_json']
    csv_write(out/'per_video.csv',runs,columns)
    csv_write(out/'differences.csv',all_diffs,['video_id','title','config','edit_type','reference','hypothesis','word_index'])
    csv_write(out/'errors.csv',errors,['video_id','title','stage','error','traceback'])
    csv_write(out/'actual_run_costs.csv',unique_cost,['video_id','title','audio_seconds',
              'api_stt_cost_usd','judge_total_cost_usd','estimated_actual_spend_this_run_usd'])
    grouped=defaultdict(list)
    for r in runs:grouped[r['config']].append(r)
    summary=[]
    for cfg in CONFIGS:
        rows=grouped[cfg]
        if not rows:continue
        N=sum(x['reference_words'] or 0 for x in rows)
        errs=sum(x['word_errors'] or 0 for x in rows)
        speakers=[(r['speaker_accuracy_aligned_percent'],r['speaker_matched_words'])
                  for r in rows if r['speaker_accuracy_aligned_percent'] is not None and r['speaker_matched_words']]
        macro=[r['wer_percent'] for r in rows if r['wer_percent'] is not None]
        cer=[r['cer_percent'] for r in rows if r['cer_percent'] is not None]
        lat=[r['parallel_latency_seconds'] for r in rows if r['parallel_latency_seconds'] is not None]
        total_words=sum((r['reference_words'] or 0) for r in rows)
        result={'config':cfg,'videos_ok':len(rows),
          'corpus_wer_percent':100*errs/N if N else None,
          'mean_video_wer_percent':statistics.mean(macro) if macro else None,
          'median_video_wer_percent':statistics.median(macro) if macro else None,
          'mean_cer_percent':statistics.mean(cer) if cer else None,
          'total_words':total_words,'total_word_errors':errs,
          'exact_match_videos':sum(r['exact_match'] or 0 for r in rows),
          'total_config_cost_usd':sum(r['estimated_cost_usd'] or 0 for r in rows),
          'mean_parallel_latency_s':statistics.mean(lat) if lat else None,
          'median_parallel_latency_s':statistics.median(lat) if lat else None,
          'speaker_accuracy_matched_percent':
              sum(a*w for a,w in speakers)/sum(w for _,w in speakers) if speakers else None,
          'speaker_scored_videos':len(speakers),
        }
        summary.append(result)
    csv_write(out/'summary.csv',summary,['config','videos_ok','corpus_wer_percent',
              'mean_video_wer_percent','median_video_wer_percent','mean_cer_percent','total_words',
              'total_word_errors','exact_match_videos','total_config_cost_usd',
              'mean_parallel_latency_s','median_parallel_latency_s',
              'speaker_accuracy_matched_percent','speaker_scored_videos'])
    evaluation={
      'summary':summary,'video_count_with_results':len(items),
      'configurations':list(CONFIGS),
      'overall_run_estimated_actual_usd':sum(x['estimated_actual_spend_this_run_usd'] for x in unique_cost),
      'notes':['Coûts = estimations selon tarifs de référence, non une facture.',
        'Coût total réel estimé = 3 moteurs ASR facturés chacun une fois par clip + 4 juges (pas la somme des sept configurations).',
        'Latence hybride = max(latences ASR individuelles) + latence juge, estimation si traitements parallèles.',
        'Speaker accuracy = score approximatif sur mots exactement alignés, après permutation des étiquettes; pas un DER.',
        'Les références peuvent contenir des erreurs; certains clips ont des textes affichés ou des voix hors référence.',
        'Le juge ne reçoit jamais la référence.']}
    (out/'report.json').write_text(json.dumps(evaluation,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report_graphics(out,runs,summary,unique_cost)
    (out/'README_RESULTATS.txt').write_text(
      'BENCHMARK LINGUATEKA - FICHIERS DE RESULTATS\n\n'
      'summary.csv : scores et latences agregees par configuration.\n'
      'per_video.csv : texte source, prediction, WER, CER, cout et latence pour chaque configuration.\n'
      'differences.csv : editions lexicales par rapport a la reference, sans ponctuation.\n'
      'errors.csv : echecs telechargement/API/jugement.\n'
      'actual_run_costs.csv : depenses estimees sans recompter les ASR partages.\n'
      'videos/*.json : transcriptions finales et metriques par video.\n'
      'cache/*/*.json : reponses brutes des fournisseurs et des juges.\n'
      'events.jsonl : journal chronologique structure.\n'
      'benchmark.log : journal lisible.\n'
      'charts/*.png : graphiques generes automatiquement.\n\n'
      'ATTENTION: La reference peut avoir des fautes. WER = (substitution + deletion + insertion) / mots reference.\n'
      'Diarisation: estimation d\'accord sur les mots alignes, ce n\'est pas un DER temporel.\n'
      'Latences hybrides: estimation parallelisee max(temps ASR) + temps arbitre.\n'
      'Couts : montants ESTIMES, pas des factures ni des debits garantis.\n'
      'Le modele juge ne voit JAMAIS les references et ne peut pas ecouter l\'audio.\n',encoding='utf-8')
    print(f'RAPPORT: {len(runs)} résultats; {len(summary)} configurations; '
          f"coût total estimé {evaluation['overall_run_estimated_actual_usd']:.4f} USD")
    return evaluation


def report_graphics(out,runs,summary,unique_cost):
    if not runs:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plots=out/'charts';plots.mkdir(exist_ok=True)
    labels=[s['config'].replace('assemblyai','Assembly').replace('elevenlabs','Eleven').replace('openai','GPT').replace('+judge','+Juge') for s in summary]
    def save(name):
        plt.tight_layout();plt.savefig(plots/name,dpi=170,bbox_inches='tight');plt.close()
    def bars(name,title,key,ylabel):
        vals=[s.get(key) if s.get(key) is not None else np.nan for s in summary]
        plt.figure(figsize=(13,5));plt.bar(range(len(vals)),vals)
        plt.xticks(range(len(vals)),labels,rotation=30,ha='right',fontsize=8)
        plt.ylabel(ylabel);plt.title(title);plt.grid(axis='y',alpha=.2)
        save(name)
    bars('01_corpus_WER.png','Taux d’erreur WER pondéré par nombre de mots','corpus_wer_percent','WER (%) - plus bas = mieux')
    bars('02_WER_moyen.png','WER moyen par Reel','mean_video_wer_percent','WER moyen (%)')
    bars('03_CER_moyen.png','Taux d’erreur au caractère','mean_cer_percent','CER moyen (%)')
    bars('04_latence_mediane.png','Latence médiane (simultanéité simulée pour les duos/trio)',
         'median_parallel_latency_s','Secondes')
    bars('05_cout_configuration.png','Coût pour générer toute la série avec chaque configuration isolément',
         'total_config_cost_usd','USD estimés')
    # Grid WER by clip and config.
    names=[]
    for r in runs:
        if r['video_id'] not in names:names.append(r['video_id'])
    matrix=np.full((len(summary),len(names)),np.nan)
    cmap={row['config']:i for i,row in enumerate(summary)}
    vmap={vid:i for i,vid in enumerate(names)}
    for r in runs:matrix[cmap[r['config']],vmap[r['video_id']]]=r['wer_percent']
    fig,ax=plt.subplots(figsize=(max(14,len(names)*.36),5.4))
    heat=ax.imshow(matrix,aspect='auto',interpolation='nearest',vmin=0,
                   vmax=np.nanpercentile(matrix,95) if np.isfinite(matrix).any() else 50)
    ax.set_xticks(range(len(names)));ax.set_xticklabels(names,rotation=75,fontsize=8)
    ax.set_yticks(range(len(summary)));ax.set_yticklabels(labels,fontsize=8)
    ax.set_xlabel('Vidéo');ax.set_title('WER (%) par vidéo et configuration');fig.colorbar(heat,ax=ax,label='WER (%)')
    save('06_heatmap_WER.png')
    # Cost/precision Pareto.
    fig,ax=plt.subplots(figsize=(9,6))
    for i,s in enumerate(summary):
        ax.scatter(s['total_config_cost_usd'],s['corpus_wer_percent'],s=70)
        ax.annotate(labels[i],(s['total_config_cost_usd'],s['corpus_wer_percent']),fontsize=8,xytext=(6,5),textcoords='offset points')
    ax.set_xlabel('Coût total estimé de la configuration (USD)');ax.set_ylabel('WER corpus (%)')
    ax.set_title('Compromis coût et précision');ax.grid(alpha=.2);save('07_compromis_cout_WER.png')
    # Distribution WER
    arrays=[[r['wer_percent'] for r in runs if r['config']==s['config'] and r['wer_percent'] is not None] for s in summary]
    plt.figure(figsize=(13,5));plt.boxplot(arrays,tick_labels=labels,showfliers=True)
    plt.xticks(rotation=30,ha='right',fontsize=8);plt.ylabel('WER par vidéo (%)');plt.title('Dispersion des erreurs');plt.grid(axis='y',alpha=.2)
    save('08_boites_WER.png')
    # End-to-end latency by video (estimated parallel in hybrids)
    fig,ax=plt.subplots(figsize=(14,6))
    for s in summary:
        d={r['video_id']:r['parallel_latency_seconds'] for r in runs if r['config']==s['config']}
        ax.plot(range(len(names)),[d.get(n,float('nan')) for n in names],marker='.',label=s['config'])
    ax.set_xticks(range(len(names)));ax.set_xticklabels(names,rotation=75,fontsize=8)
    ax.set_ylabel('Secondes');ax.set_title('Latence par vidéo');ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.2)
    save('09_latence_par_video.png')
    # Incurred total estimated billed costs (unique requests)
    costs=[r['estimated_actual_spend_this_run_usd'] for r in unique_cost]
    fig,ax=plt.subplots(figsize=(13,5));ax.plot(range(1,len(costs)+1),np.cumsum(costs),marker='o',ms=3)
    ax.set_ylabel('USD');ax.set_xlabel('Nombre de vidéos terminées')
    ax.set_title('Coût cumulé du benchmark (3 ASR uniques + 4 juges)');ax.grid(alpha=.2)
    save('10_cout_cumule.png')
    # Diarization aligned word accuracy.
    sp=[s['speaker_accuracy_matched_percent'] if s.get('speaker_accuracy_matched_percent') is not None else np.nan for s in summary]
    fig,ax=plt.subplots(figsize=(13,5));ax.bar(range(len(sp)),sp)
    ax.set_ylim(0,105);ax.set_xticks(range(len(labels)));ax.set_xticklabels(labels,rotation=30,ha='right',fontsize=8)
    ax.set_ylabel('Accord locuteurs (%) sur mots alignés');ax.set_title('Diarisation (proxy, pas DER)');ax.grid(axis='y',alpha=.2)
    save('11_accord_locuteurs.png')
    # Winner count by video and tied cases.
    wins=Counter()
    for name in names:
        present=[r for r in runs if r['video_id']==name and r['wer_percent'] is not None]
        if not present:continue
        best=min(r['wer_percent'] for r in present)
        bests=[r['config'] for r in present if abs(r['wer_percent']-best)<1e-9]
        if len(bests)>1:wins['ex aequo']+=1
        else:wins[bests[0]]+=1
    fig,ax=plt.subplots(figsize=(12,5))
    wk=list(wins);ax.bar(range(len(wk)),[wins[k] for k in wk]);ax.set_xticks(range(len(wk)))
    ax.set_xticklabels(wk,rotation=30,ha='right',fontsize=8)
    ax.set_ylabel('Vidéos gagnées');ax.set_title('Configuration au WER minimum par Reel');ax.grid(axis='y',alpha=.2)
    save('12_gagnants_par_video.png')
    # Judge improvement vs best of its source ASR models.
    fig,ax=plt.subplots(figsize=(12,5))
    hybrid=[k for k in CONFIGS if '+judge' in k]
    gains=[]
    for key in hybrid:
        diff=[]
        sources=CONFIGS[key]
        for video in names:
            pred=[r for r in runs if r['video_id']==video and r['config']==key]
            solo=[r for r in runs if r['video_id']==video and r['config'] in sources]
            if pred and len(solo)==len(sources):
                diff.append(min(r['wer_percent'] for r in solo)-pred[0]['wer_percent'])
        gains.append(statistics.mean(diff) if diff else float('nan'))
    ax.bar(range(len(hybrid)),gains);ax.axhline(0,linewidth=.8)
    ax.set_xticks(range(len(hybrid)));ax.set_xticklabels(hybrid,rotation=20,ha='right',fontsize=8)
    ax.set_ylabel('Gain moyen WER en points (positif = juge meilleur que meilleure IA du duo/trio)')
    ax.set_title('Le juge améliore-t-il vraiment le meilleur modèle disponible ?')
    ax.grid(axis='y',alpha=.2);save('13_gain_juge_meilleur_ASR.png')
    # Error types total
    fig,ax=plt.subplots(figsize=(13,5));x=np.arange(len(summary));bottom=np.zeros(len(summary))
    for typ in ('substitutions','deletions','insertions'):
        values=np.array([sum(r[typ] for r in runs if r['config']==s['config']) for s in summary])
        ax.bar(x,values,bottom=bottom,label=typ);bottom+=values
    ax.set_xticks(x);ax.set_xticklabels(labels,rotation=30,ha='right',fontsize=8)
    ax.set_ylabel('Nombre total');ax.set_title('Répartition des erreurs de mots');ax.legend()
    save('14_types_erreurs.png')
    # WER by clip length
    fig,ax=plt.subplots(figsize=(10,6))
    for s in summary:
        pairs=[r for r in runs if r['config']==s['config'] and r['audio_seconds'] and r['wer_percent'] is not None]
        ax.scatter([r['audio_seconds'] for r in pairs],[r['wer_percent'] for r in pairs],s=22,label=s['config'],alpha=.55)
    ax.set_xlabel('Durée audio (s)');ax.set_ylabel('WER par clip (%)')
    ax.set_title('Les vidéos courtes sont-elles plus difficiles ?');ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.2)
    save('15_longueur_vs_WER.png')
    # Single-provider latency vs clip length.
    fig,ax=plt.subplots(figsize=(10,6))
    for cfg in ('openai','assemblyai','elevenlabs'):
        subset=[r for r in runs if r['config']==cfg and r['audio_seconds']]
        ax.scatter([r['audio_seconds'] for r in subset],[r['parallel_latency_seconds'] for r in subset],label=cfg)
    ax.set_xlabel('Durée audio (s)');ax.set_ylabel('Latence réelle ASR (s)')
    ax.set_title('Performance des trois moteurs en fonction de la durée');ax.legend();ax.grid(alpha=.2)
    save('16_longueur_vs_latence.png')
    # Judge costs separately.
    fig,ax=plt.subplots(figsize=(12,5))
    vals=[sum(r['judge_cost_usd'] for r in runs if r['config']==cfg) for cfg in hybrid]
    ax.bar(range(len(hybrid)),vals);ax.set_xticks(range(len(hybrid)))
    ax.set_xticklabels(hybrid,rotation=20,ha='right',fontsize=8)
    ax.set_ylabel('USD estimés');ax.set_title('Coût additionnel de l’arbitrage textuel')
    save('17_cout_juge.png')

if __name__=='__main__':
    location=Path(sys.argv[1] if len(sys.argv)>1 else 'results')
    print(json.dumps(build_report(location),ensure_ascii=False,indent=2))
