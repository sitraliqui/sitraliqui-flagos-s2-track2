"""Recompute manifest, package identity and raw benchmark metrics offline."""
import hashlib,json,zipfile
from pathlib import Path,PurePosixPath
from accuracy_gate import check as accuracy_check
from csv_validator import validate_benchmark
ROOT=Path(__file__).resolve().parents[1]

def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def digest(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(value,message):
 if not value:raise ValueError(message)

def performance(root,script,cases):
 raw=list(root.glob('**/raw_runs_*.csv')) or list(root.glob('raw.csv'))
 summary=list(root.glob('**/summary_*.csv')) or list(root.glob('summary.csv'))
 require(len(raw)==len(summary)==1,'Exactly one raw/summary CSV required: '+str(root))
 return validate_benchmark(raw[0],summary[0],script,cases)

def main():
 manifest=load(ROOT/'manifest.json')
 actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()}
 require(actual==set(manifest)|{'manifest.json'},'Missing or extra archive files; run with -B')
 for name,row in manifest.items():
  require(not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts,'Unsafe manifest path')
  p=ROOT/name
  require(p.resolve().is_relative_to(ROOT) and not p.is_symlink(),'Unsafe path '+name)
  require(p.stat().st_size==row['bytes'] and digest(p)==row['sha256'],'Changed file '+name)
 builds=load(ROOT/'provenance/build-and-install.json')['builds']
 for row in builds:
  wheel=ROOT/row['wheel'];require(digest(wheel)==row['sha256'],'Wrong wheel')
  pkg='vllm_fl' if row['project']=='vllm-plugin-FL' else 'flag_gems'
  base=ROOT/row['project']/('' if pkg=='vllm_fl' else 'src')
  with zipfile.ZipFile(wheel) as z:
   require(z.testzip() is None,'Corrupt wheel')
   for name in z.namelist():
    if name.startswith(pkg+'/') and not name.endswith('/'):
     require((base/name).read_bytes()==z.read(name),'Source/wheel mismatch '+name)
 def payload(path,package):
  with zipfile.ZipFile(path) as archive:
   require(archive.testzip() is None,'Corrupt historical wheel')
   return {name:archive.read(name) for name in archive.namelist() if name.startswith(package+'/') and not name.endswith('/')}
 for audit in load(ROOT/'provenance/tianshu-before-after.json'):
  package='vllm_fl' if audit['project']=='vllm-plugin-FL' else 'flag_gems'
  previous=ROOT/'baselines/tianshu-formal4'/audit['reference_wheel']
  require(digest(previous)==audit['reference_sha256'],'Historical TianShu wheel identity differs')
  left=payload(previous,package);right=payload(ROOT/'wheels/validated'/audit['final_wheel'],package)
  changes=sorted(name for name in left.keys()|right.keys() if left.get(name)!=right.get(name))
  require(changes==sorted(row['path'] for row in audit['changed_members']),'TianShu before/after difference declaration differs')
  allowed={package+'/_version.py'}
  if package=='vllm_fl':allowed|={'vllm_fl/dispatch/backends/vendor/metax/impl/attention/flash_attn.py','vllm_fl/configs/moe/E=256,N=512,device_name=nvidia.json'}
  require(set(changes)<=allowed,'Unexpected TianShu runtime source difference')
 control=load(ROOT/'baselines/default-prefix-control.json')
 control_wheel=ROOT/'baselines'/control['wheel']
 require(digest(control_wheel)==control['sha256'],'Default control wheel identity differs')
 current_plugin=next(r for r in builds if r['project']=='vllm-plugin-FL')
 left=payload(control_wheel,'vllm_fl');right=payload(ROOT/current_plugin['wheel'],'vllm_fl')
 target='vllm_fl/dispatch/backends/vendor/metax/impl/attention/flash_attn.py'
 changed={name for name in left.keys()|right.keys() if left.get(name)!=right.get(name)}
 require(changed=={'vllm_fl/_version.py',target},'Default control has unrelated runtime differences')
 require(hashlib.sha256(left[target]).hexdigest()==control['restored_source_sha256'],'Default prefix source is not the archived original')
 script=ROOT/'vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py'
 receipt=load(ROOT/'evidence/acceptance.json')
 scores={}
 for label,folder,cases in [('tianshu_formal4','tianshu-formal4',[('4k',('4k',)),('16k',('16k',))]),('metax_historical_stock','metax-stock',[('performance',('4k','16k'))])]:
  base=ROOT/'evidence'/folder
  a=accuracy_check(base/'accuracy')
  require(a['correct']==receipt[label]['accuracy']['correct'],label+' accuracy differs')
  perf={}
  for sub,names in cases:
   r=performance(base/sub,script,names)
   for case in r['cases']:
    expected=receipt[label]['performance'][case['case']]
    require(case['summary']['Total tok/s']==expected['total_tokens_per_second'],label+' throughput differs')
    require(case['summary']['Mean TTFT (ms)']==expected['mean_ttft_ms'],label+' TTFT differs')
    perf[case['case']]=case['summary']
  scores[label]=dict(accuracy=a,performance=perf)
 old=ROOT/'evidence/metax-prefixb-105-original'
 scores['metax_prefixb_original']=dict(accuracy=accuracy_check(old/'eval'),
  performance=performance(ROOT/'evidence/metax-prefixb-performance-original/prefixb',script,('4k','16k')))
 final=ROOT/'evidence/metax-finalb1'
 if final.exists():
  status=load(final/'status.json')
  require(status['stage']==receipt['metax_merged']['stage'],'Merged status differs')
  if (final/'accuracy-audit.json').exists():
   scores['metax_merged_accuracy']=accuracy_check(final/'accuracy')
   require(scores['metax_merged_accuracy']['correct']==receipt['metax_merged']['accuracy']['correct'],'Merged accuracy claim differs from raw reviews')
  if (final/'performance-audit.json').exists():
   scores['metax_merged_performance']=performance(final/'performance',script,('4k','16k'))
   for case in scores['metax_merged_performance']['cases']:
    claimed=receipt['metax_merged']['performance'][case['case']]
    require(case['summary']['Total tok/s']==claimed['total_tokens_per_second'] and case['summary']['Mean TTFT (ms)']==claimed['mean_ttft_ms'],'Merged metrics differ from raw CSV')
  if status['stage']=='completed':
   guardian=load(final/'guardian/receipt.json')
   require(guardian['exit_code']==0 and guardian['cleanup_complete'] and not guardian['survivors'],'Unclean final run')
   installed=json.loads((final/'installation-audit-formal.log').read_text())
   require({x['wheel_sha256'] for x in installed}=={x['sha256'] for x in builds},'Official run package identity differs')
   require(json.loads((final/'installation-audit-after.log').read_text())==installed,'Installed package changed during formal run')
   identity=load(final/'installed-prefix-source.json')
   require(identity['diagnostic_marker_absent'] and identity['sha256']==receipt['metax_merged']['prefix_source_sha256'],'Diagnostic code remained installed or source changed')
   hit=load(final/'hit-source.json')
   require(hit['num_prefills']>0 and hit['diagnostic_only'],'Missing diagnostic hit')
 default=ROOT/'evidence/metax-matched-default'
 if default.exists():
  status=load(default/'status.json')
  require(status['stage']==receipt['metax_matched_default']['stage'],'Default control status differs')
  if (default/'accuracy-audit.json').exists():
   scores['metax_matched_default_accuracy']=accuracy_check(default/'accuracy')
   require(scores['metax_matched_default_accuracy']['correct']==receipt['metax_matched_default']['accuracy']['correct'],'Default accuracy claim differs')
  if (default/'performance-audit.json').exists():
   scores['metax_matched_default_performance']=performance(default/'performance',script,('4k','16k'))
   for case in scores['metax_matched_default_performance']['cases']:
    claimed=receipt['metax_matched_default']['performance'][case['case']]
    require(case['summary']['Total tok/s']==claimed['total_tokens_per_second'] and case['summary']['Mean TTFT (ms)']==claimed['mean_ttft_ms'],'Default control metrics differ')
  if status['stage']=='completed':
   guardian=load(default/'guardian/receipt.json')
   require(guardian['exit_code']==0 and guardian['cleanup_complete'] and not guardian['survivors'],'Unclean default control')
   installed=json.loads((default/'installation-audit-formal.log').read_text())
   expected={receipt['metax_matched_default']['wheel_sha256'],next(r['sha256'] for r in builds if r['project']=='FlagGems')}
   require({x['wheel_sha256'] for x in installed}==expected,'Default control installed wrong packages')
   require(json.loads((default/'installation-audit-after.log').read_text())==installed,'Default control install changed')
   identity=load(default/'installed-prefix-source.json')
   require(identity['diagnostic_marker_absent'] and identity['sha256']==control['restored_source_sha256'],'Default control source changed')
 references=load(ROOT/'provenance/official-rules-reference.json')['references']['metax']
 repeat=ROOT/'evidence/metax-4k-verification'
 if repeat.exists():
  status=load(repeat/'status.json')
  claimed=receipt['metax_4k_verification']
  require(status['stage']==claimed['stage']=='completed','Repeat did not complete')
  require(not claimed['new_accuracy_run'] and not claimed['primary_result_replaced'],'Repeat misrepresented as a fresh accuracy run or replacement')
  scores['metax_4k_verification']=performance(repeat/'performance',script,('4k',))
  summary=scores['metax_4k_verification']['cases'][0]['summary']
  require(summary['Total tok/s']==claimed['performance']['4k']['total_tokens_per_second'] and summary['Mean TTFT (ms)']==claimed['performance']['4k']['mean_ttft_ms'],'Repeat metrics differ')
  installed=json.loads((repeat/'installation-audit-formal.log').read_text())
  require({x['wheel_sha256'] for x in installed}=={x['sha256'] for x in builds},'Repeat candidate changed')
  require(json.loads((repeat/'installation-audit-after.log').read_text())==installed,'Repeat install changed')
  identity=load(repeat/'installed-prefix-source.json')
  require(identity['diagnostic_marker_absent'] and identity['sha256']==receipt['metax_merged']['prefix_source_sha256'],'Repeat source changed')
  guardian=load(repeat/'guardian/receipt.json')
  require(guardian['exit_code']==0 and guardian['cleanup_complete'] and not guardian['survivors'],'Unclean repeat')
 for case,row in receipt['metax_merged']['performance'].items():
  ref=references[case];claimed=receipt['metax_merged']['public_reference_comparison'][case]
  require(claimed['throughput_within_reference_tolerance']==(row['total_tokens_per_second']>=.99*ref['total_tokens_per_second']),'Incorrect public throughput tolerance claim')
  require(claimed['TTFT_within_reference_tolerance']==(row['mean_ttft_ms']<=1.01*ref['mean_ttft_ms']),'Incorrect public TTFT tolerance claim')
 require(not receipt['submission_authorized'],'Unexpected submission authorization')
 closeout=ROOT/'evidence/resource-closeout.json'
 if closeout.exists():
  resource=load(closeout)
  require(resource['platform_state']=='OFF' and resource['remaining_phase_CNY']>=0,'Resource not closed or budget exceeded')
 print(json.dumps(dict(archive_integrity='passed',files=len(manifest),metrics_recomputed=scores,
  limitations=['No claim of TianShu merged formal retest','Historical dates are not a controlled speedup denominator','No competition upload performed']),ensure_ascii=False,indent=2))
if __name__=='__main__':main()

