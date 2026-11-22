import json, os, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import benchmark
import report

class UnitTests(unittest.TestCase):
    def test_word_metrics(self):
        a=benchmark.ed_stats('A: Turystę spotkałam.','Turysta spotkałam!')
        self.assertEqual(a['reference_words'],2)
        self.assertEqual(a['substitutions'],1)
        self.assertEqual(a['word_errors'],1)
        self.assertAlmostEqual(a['wer'],.5)
    def test_all_equal(self):
        x=benchmark.ed_stats('A: No, siemanko!','No siemanko')
        self.assertEqual(x['wer'],0)
        self.assertEqual(x['cer'],0)
    def test_mismatch_insertion(self):
        x=benchmark.ed_stats('A: Idziemy!', 'Idziemy teraz')
        self.assertEqual(x['insertions'],1)
    def test_speaker_permutation(self):
        source='A: Cześć wszystkim.\nB: No tak, idziemy.'
        pred=[{'speaker':'B','text':'Cześć wszystkim.'},{'speaker':'A','text':'No tak, idziemy.'}]
        x=benchmark.diarization_agreement(source,pred)
        self.assertAlmostEqual(x['speaker_accuracy_aligned'],1.0)
    def test_actual_37_clips(self):
        p=Path(__file__).resolve().parents[1]/'dataset.json'
        d=benchmark.load_json(p)
        self.assertEqual(len(d['videos']),37)
        self.assertEqual(len(set(x['url'] for x in d['videos'])),37)
        self.assertTrue(all(x['reference_text'] for x in d['videos']))
    def test_end_to_end_offline_mock_seven_configs(self):
        doc={'id':'900','title':'Synthetic audio test','url':'https://www.instagram.com/p/FAKE/',
             'reference_text':'A: Idziemy!\nB: Turystę spotkałam.'}
        calls=[]
        def fake_audio(url, tmp):
            audio=Path(tmp)/'audio.mp3';audio.write_bytes(b'FAKE_MP3_DATA')
            return audio,60.0,.2,.1
        def fake_stt(provider):
            def impl(audio):
                calls.append(provider)
                return {'text':'Idziemy! Turystę spotkałam.',
                   'turns':[] if provider=='openai' else [
                       {'speaker':'A','text':'Idziemy!'},
                       {'speaker':'B','text':'Turystę spotkałam.'}],
                   'words':[],'model_used':provider,'latency_s':.8,'provider_payload':{'status':'fake'}}
            return impl
        def fake_judge(providers,outputs):
            calls.append('judge')
            return {'text':'Idziemy! Turystę spotkałam.',
                    'turns':[{'speaker':'A','text':'Idziemy!'},{'speaker':'B','text':'Turystę spotkałam.'}],
                    'judge_usage':{'input_tokens':150,'output_tokens':70,'estimated_usd':0.0002},
                    'judge_latency_s':.4,'judge_model':'mock','uncertainties':[]}
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)
            with patch.object(benchmark,'download_and_extract',side_effect=fake_audio), \
                 patch.object(benchmark,'transcribe_openai',side_effect=fake_stt('openai')), \
                 patch.object(benchmark,'transcribe_assemblyai',side_effect=fake_stt('assemblyai')), \
                 patch.object(benchmark,'transcribe_elevenlabs',side_effect=fake_stt('elevenlabs')), \
                 patch.object(benchmark,'judge_transcripts',side_effect=fake_judge):
                result=benchmark.run_video(doc,out)
                self.assertEqual(len(result['configurations']),7)
                self.assertEqual(len(calls),7)
                self.assertEqual(len([x for x in calls if x=='judge']),4)
                # Cache (same video) must not generate new billable requests.
                again=benchmark.run_video(doc,out)
                self.assertEqual(len(calls),7)
                self.assertEqual(len(again['configurations']),7)
            stats=report.build_report(out)
            self.assertEqual(len(stats['summary']),7)
            self.assertGreaterEqual(len(list((out/'charts').glob('*.png'))),16)
            self.assertTrue((out/'per_video.csv').exists())
            self.assertTrue((out/'events.jsonl').exists())

if __name__=='__main__':unittest.main()
